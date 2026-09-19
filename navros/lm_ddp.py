"""NAVROS-LM en varias GPU (pensado para 2×T4 de Kaggle, 15 GB cada una).

Por qué así y no con FSDP:
  * Pesos fp32 replicados + DDP para promediar gradientes (una all-reduce por paso).
  * El optimizador se reparte POR DUEÑO (estilo ZeRO-1): cada matriz pertenece a un rango,
    que guarda su estado (momento de Muon o m/v de Adam), la ortogonaliza con Newton-Schulz
    y después la difunde a los demás. Muon necesita la matriz completa para ortogonalizar,
    así que repartir por matrices enteras evita juntar trozos y no duplica trabajo.
  * fp16 con escalado dinámico de la pérdida (la T4 no tiene bf16). La decisión de saltar un
    paso se toma sobre los gradientes ya promediados, idénticos en todos los rangos, así que
    todos saltan o aplican a la vez.
  * El corte por tiempo lo decide el rango 0 y lo difunde: todos paran en el mismo paso.
  * Checkpoint: el rango 0 guarda pesos, datos y escalador; cada rango guarda su parte del
    optimizador. Reanudar es exacto porque el reparto de dueños es determinista.

Se lanza con torchrun:  python -m torch.distributed.run --standalone --nproc_per_node=2 scripts/05_train_lm.py --json '{...}'
"""
from __future__ import annotations

import contextlib
import glob
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from .backend import Backend
from .lm import LMRun, eval_losses, eval_recurrence, lm_preset, lr_mult, make_data, step_r, train_flops_per_token
from .pt.model import Navros
from .pt.optim import Muon


def partition(named, world):
    """Reparto voraz por tamaño: cada parámetro entero a un rango. Determinista."""
    loads, owner = [0] * world, {}
    for n, p in sorted(named, key=lambda x: (-x[1].numel(), x[0])):
        r = loads.index(min(loads))
        owner[n] = r
        loads[r] += p.numel()
    return owner, loads


class LossScaler:
    """Escalado dinámico de la pérdida para fp16, sobre gradientes ya promediados entre rangos."""

    def __init__(self, init=2.0 ** 15, growth_interval=500, max_scale=2.0 ** 24):
        self.scale, self.interval, self.max, self.good, self.skipped = init, growth_interval, max_scale, 0, 0

    @torch.no_grad()
    def unscale_clip(self, params, clip):
        """Devuelve (ok, norma sin escalar). Si ok, deja los gradientes desescalados y recortados."""
        grads = [p.grad for p in params if p.grad is not None]
        total = torch.sqrt(sum((g.float() ** 2).sum() for g in grads))
        tv = float(total)
        if not math.isfinite(tv):
            self.scale /= 2.0
            self.good = 0
            self.skipped += 1
            return False, tv
        norm = tv / self.scale
        coef = (min(1.0, clip / (norm + 1e-6)) if clip else 1.0) / self.scale
        for g in grads:
            g.mul_(coef)
        self.good += 1
        if self.good % self.interval == 0:
            self.scale = min(self.scale * 2.0, self.max)
        return True, norm

    def state_dict(self):
        return dict(scale=self.scale, good=self.good, skipped=self.skipped)

    def load_state_dict(self, st):
        self.scale, self.good, self.skipped = st["scale"], st["good"], st["skipped"]


def find_checkpoint(rc: LMRun):
    """Checkpoint más avanzado (propio o adjunto como entrada) de este preset y etiqueta."""
    cands = []
    pats = ([f"{rc.ckpt_dir}/latest.json"] if rc.ckpt_dir else []) + ["/kaggle/input/**/latest.json"]
    for pat in pats:
        for p in glob.glob(pat, recursive=True):
            try:
                meta = json.loads(Path(p).read_text())
                if meta.get("preset") == rc.preset and meta.get("tag") == rc.tag and meta.get("kind") == "ddp":
                    cands.append((meta["step"], str(Path(p).parent)))
            except Exception:
                pass
    return max(cands)[1] if cands else None


def train_ddp(rc: LMRun, ns_dtype: str | None = "fp16", log=print) -> dict:
    t_start = time.time()
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local = int(os.environ.get("LOCAL_RANK", 0))
    use_cuda = rc.device.startswith("cuda") and torch.cuda.is_available()
    if world > 1 and not dist.is_initialized():
        from datetime import timedelta
        dist.init_process_group("nccl" if use_cuda else "gloo", timeout=timedelta(hours=1))
    if use_cuda:
        torch.cuda.set_device(local)
    device = torch.device(f"cuda:{local}" if use_cuda else "cpu")
    say = log if rank == 0 else (lambda s: None)
    be = Backend(str(device), rc.precision)
    fp16 = rc.precision == "fp16"

    data = make_data(rc)
    cfg = lm_preset(rc.preset, data.vocab)
    torch.manual_seed(rc.seed)
    model = Navros(cfg).to(device)
    if rc.grad_ckpt:
        model.ckpt_fn = be.ckpt_fn()
    named = list(model.named_parameters())
    owner, loads = partition(named, world)
    mine = [(n, p) for n, p in named if owner[n] == rank]
    nsd = {"fp16": torch.float16, "bf16": torch.bfloat16, None: None, "fp32": None}[ns_dtype]
    opt = Muon(mine, lr_muon=rc.lr_muon, lr_adam=rc.lr_adam, wd_muon=rc.wd, ns_dtype=nsd if use_cuda else None,
               buf_dtype=torch.bfloat16 if rc.muon_buf == "bf16" else None)
    ddp = DDP(model, device_ids=[local] if use_cuda else None, gradient_as_bucket_view=True,
              broadcast_buffers=False) if world > 1 else model
    scaler = LossScaler() if fp16 else None
    evals = {l: data.eval_set(l, "val", rc.eval_seq) for l in rc.langs} if rank == 0 else {}

    step, history, extra = 0, [], {}
    ck = find_checkpoint(rc)
    if ck:
        st = torch.load(f"{ck}/model.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        data.load_state(st["data"])
        if scaler is not None and st["scaler"]:
            scaler.load_state_dict(st["scaler"])
        step, extra = st["step"], st["extra"]
        history = extra.get("history", [])
        so = torch.load(f"{ck}/opt_r{rank}.pt", map_location="cpu", weights_only=False)
        opt.load_state_dict(so["opt"])
        opt.t = so["t"]
        say(f"reanudado desde {ck} en el paso {step}")

    n_micro = rc.batch // (rc.micro * world)
    assert n_micro * rc.micro * world == rc.batch, "batch debe ser múltiplo de micro × world"
    r_bar = round(cfg.r_mean) if cfg.recurrent else None
    flops_tok = train_flops_per_token(cfg, cfg.r_mean if cfg.recurrent else 0,
                                      min(cfg.k_bptt, cfg.r_mean) if cfg.recurrent else 0, rc.T)
    say(f"{rc.preset}: {sum(p.numel() for p in model.parameters()):,} params · {world} rangos · reparto "
        f"{[round(x / 1e6) for x in loads]}M · {rc.steps} pasos de {rc.batch * rc.T:,} tokens · "
        f"{flops_tok / 1e9:.2f} GFLOP/token")

    def save(tag_step):
        if not rc.ckpt_dir:
            return
        d = Path(rc.ckpt_dir)
        if rank == 0:
            d.mkdir(parents=True, exist_ok=True)
        if world > 1:
            dist.barrier()
        torch.save(dict(opt=opt.state_dict(), t=opt.t), d / f"opt_r{rank}.tmp")
        if rank == 0:
            torch.save(dict(step=tag_step, model=model.state_dict(), data=data.state(),
                            scaler=scaler.state_dict() if scaler else None, extra=dict(history=history)), d / "model.tmp")
        if world > 1:
            dist.barrier()
        os.replace(d / f"opt_r{rank}.tmp", d / f"opt_r{rank}.pt")
        if rank == 0:
            os.replace(d / "model.tmp", d / "model.pt")
        if world > 1:
            dist.barrier()
        if rank == 0:
            (d / "latest.json").write_text(json.dumps(dict(step=tag_step, preset=rc.preset, tag=rc.tag, kind="ddp",
                                                           world=world, time=time.time())))

    last_ck, t_log, tok_log, run, n_run = time.time(), time.time(), 0, 0.0, 0
    flag = torch.zeros(1, device=device)
    while step < rc.steps:
        flag.fill_(1.0 if (rank == 0 and time.time() - t_start > rc.time_budget_s) else 0.0)
        if world > 1:
            dist.broadcast(flag, 0)
        if flag.item() > 0:
            say("presupuesto de tiempo agotado: checkpoint y salida")
            break
        r, k = step_r(rc, cfg, step, be)
        model.zero_grad(set_to_none=False)
        S = scaler.scale if scaler else 1.0
        tot = 0.0
        for i in range(n_micro):
            xa, ya = data.batch(rc.micro * world)  # todos los rangos avanzan el mismo cursor
            x = xa[rank * rc.micro:(rank + 1) * rc.micro].to(device, non_blocking=True)
            y = ya[rank * rc.micro:(rank + 1) * rc.micro].to(device, non_blocking=True)
            sync = contextlib.nullcontext() if (world == 1 or i == n_micro - 1) else ddp.no_sync()
            with sync:
                with torch.autocast(device.type, dtype=be.dtype, enabled=be.dtype != torch.float32, cache_enabled=False):
                    logits = ddp(x, r=r, k=k)
                loss = F.cross_entropy(logits.float().flatten(0, 1), y.flatten()) / n_micro
                (loss * S).backward()
            tot += float(loss.detach())
        params = [p for _, p in named]
        if scaler:
            ok, gn = scaler.unscale_clip(params, rc.clip)
        else:
            gn = float(torch.sqrt(sum((p.grad.float() ** 2).sum() for p in params)))
            ok = math.isfinite(gn)
            if ok and rc.clip:
                c = min(1.0, rc.clip / (gn + 1e-6))
                for p in params:
                    p.grad.mul_(c)
        if ok:
            opt.step(lr_mult(step, rc))
            if world > 1:
                with torch.no_grad():
                    for n, p in named:
                        dist.broadcast(p.data, src=owner[n])
        step += 1
        run += tot
        n_run += 1
        tok_log += rc.batch * rc.T
        if step % rc.log_every == 0:
            dt = time.time() - t_log
            row = dict(step=step, loss=run / n_run, gnorm=gn, tok_s=tok_log / dt, tflops=tok_log * flops_tok / dt / 1e12,
                       lr=lr_mult(step, rc), scale=scaler.scale if scaler else None,
                       skipped=scaler.skipped if scaler else 0, elapsed=time.time() - t_start,
                       mem_gb=torch.cuda.max_memory_allocated(device) / 1e9 if use_cuda else None)
            history.append(row)
            say(f"[{rc.tag or rc.preset}] paso {step}/{rc.steps} loss {row['loss']:.4f} |g| {gn:.2f} "
                f"{row['tok_s'] / 1e3:.2f}K tok/s {row['tflops']:.1f} TFLOP/s escala {row['scale']} "
                f"saltados {row['skipped']} mem {row['mem_gb']:.1f} GB" if use_cuda else
                f"[{rc.tag or rc.preset}] paso {step}/{rc.steps} loss {row['loss']:.4f}")
            t_log, tok_log, run, n_run = time.time(), 0, 0.0, 0
        if step % rc.eval_every == 0 or step == rc.steps:
            if rank == 0:
                ev = {l: eval_losses(model, evals[l], rc, be, r=r_bar) for l in rc.langs}
                history.append(dict(step=step, val=ev))
                say(f"[{rc.tag or rc.preset}] paso {step} val " + " ".join(f"{l} {v:.4f}" for l, v in ev.items()))
            if world > 1:
                dist.barrier()
        if time.time() - last_ck > rc.ckpt_every_min * 60:
            save(step)
            last_ck = time.time()
    save(step)
    done = step >= rc.steps
    res = dict(run=asdict(rc), model=cfg.to_dict(), params=sum(p.numel() for p in model.parameters()), world=world,
               flops_per_token=flops_tok, step=step, finished=done, history=history, seconds=time.time() - t_start,
               skipped=scaler.skipped if scaler else 0)
    if rank == 0 and done:
        res["val"] = {l: eval_losses(model, evals[l], rc, be, r=r_bar) for l in rc.langs}
        tests = {l: data.eval_set(l, "test", rc.eval_seq) for l in rc.langs}
        res["test"] = {l: eval_losses(model, tests[l], rc, be, r=r_bar) for l in rc.langs}
        if cfg.recurrent:
            res["recurrence"] = {l: eval_recurrence(model, evals[l], rc, be) for l in rc.langs}
    if world > 1:
        dist.barrier()
    return res
