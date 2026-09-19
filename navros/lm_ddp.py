"""NAVROS-LM en varias GPU (pensado para 2×T4 de Kaggle, 14,5 GB útiles cada una).

Precisión mixta clásica (estilo Megatron/DeepSpeed ZeRO-1), sin DDP:
  * Cada GPU tiene el modelo completo en fp16 (o bf16) y calcula gradientes en fp16.
  * Cada matriz tiene un rango DUEÑO, que guarda su copia maestra fp32 y su estado de
    optimizador (Muon necesita la matriz entera para ortogonalizar: se reparte por matrices
    enteras, así no hay que juntar trozos ni se duplica trabajo).
  * Tras acumular los micro-lotes, cada gradiente se REDUCE en fp32 solo hacia su dueño y el
    resto lo libera. La norma global (para detectar inf/NaN del escalado de fp16 y para el
    recorte) se obtiene sumando un escalar entre rangos: la decisión es idéntica en todos.
  * El dueño aplica el paso sobre su maestro fp32, lo copia a la versión fp16 y la difunde.
  * El corte por tiempo lo decide el rango 0 y lo difunde: todos paran en el mismo paso.
  * Checkpoint: cada rango guarda sus maestros fp32 y su estado de optimizador; el rango 0,
    además, datos, escalador e historial. Reanudar es exacto (reparto determinista).

Memoria por GPU con NAVROS-1B: 2,1 GB (modelo fp16) + 2,1 (gradientes fp16) + 2,1 (maestros
fp32 propios) + ~1 (momento de Muon bf16 propio) + activaciones con recomputación.

Lanzamiento sin torchrun: navros/launch.py (un proceso por GPU).
"""
from __future__ import annotations

import glob
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

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


def train_ddp(rc: LMRun, ns_dtype: str | None = "fp16", log=print, on_save=None) -> dict:
    """on_save(): se llama tras cada checkpoint (p. ej. para confirmar un Volume de Modal)."""
    t_start = time.time()
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local = int(os.environ.get("LOCAL_RANK", 0))
    use_cuda = rc.device.startswith("cuda") and torch.cuda.is_available()
    if world > 1 and not dist.is_initialized():
        from datetime import timedelta
        dist.init_process_group("nccl" if use_cuda else "gloo", timeout=timedelta(hours=1))
    if world == 1 and use_cuda and ":" in rc.device:   # un solo proceso en una GPU concreta (rejillas)
        device = torch.device(rc.device)
    else:
        device = torch.device(f"cuda:{local}" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(device)
    say = log if rank == 0 else (lambda s: None)
    be = Backend(str(device), rc.precision)
    low = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[rc.precision]
    fp16 = rc.precision == "fp16"

    data = make_data(rc)
    cfg = lm_preset(rc.preset, data.vocab)
    torch.manual_seed(rc.seed)
    model = Navros(cfg)                                   # init fp32 idéntica en todos los rangos (misma semilla)
    named = list(model.named_parameters())
    owner, loads = partition(named, world)
    masters = {n: torch.nn.Parameter(p.detach().clone().float().to(device)) for n, p in named if owner[n] == rank}
    model.to(device=device, dtype=low)                    # copia de trabajo en baja precisión
    if rc.grad_ckpt:
        model.ckpt_fn = be.ckpt_fn()
    named = list(model.named_parameters())
    nsd = {"fp16": torch.float16, "bf16": torch.bfloat16, None: None, "fp32": None}[ns_dtype]
    opt = Muon(list(masters.items()), lr_muon=rc.lr_muon, lr_adam=rc.lr_adam, wd_muon=rc.wd,
               ns_dtype=nsd if use_cuda else None, buf_dtype=torch.bfloat16 if rc.muon_buf == "bf16" else None)
    scaler = LossScaler() if fp16 else None
    evals = {l: data.eval_set(l, "val", rc.eval_seq) for l in rc.langs} if rank == 0 else {}

    @torch.no_grad()
    def publish():
        """Maestros fp32 → copia de trabajo, y difusión desde cada dueño."""
        for n, p in named:
            if owner[n] == rank:
                p.copy_(masters[n].to(p.dtype))
            if world > 1:
                dist.broadcast(p.data, src=owner[n])

    step, history, extra = 0, [], {}
    ck = find_checkpoint(rc)
    if ck:
        st = torch.load(f"{ck}/model.pt", map_location="cpu", weights_only=False)
        data.load_state(st["data"])
        if scaler is not None and st["scaler"]:
            scaler.load_state_dict(st["scaler"])
        step, extra = st["step"], st["extra"]
        history = extra.get("history", [])
        so = torch.load(f"{ck}/opt_r{rank}.pt", map_location="cpu", weights_only=False)
        with torch.no_grad():
            for n, t in so["masters"].items():
                masters[n].copy_(t)
        opt.load_state_dict(so["opt"])
        opt.t = so["t"]
        say(f"reanudado desde {ck} en el paso {step}")
    publish()

    n_micro = rc.batch // (rc.micro * world)
    assert n_micro * rc.micro * world == rc.batch, "batch debe ser múltiplo de micro × world"
    r_bar = round(cfg.r_mean) if cfg.recurrent else None
    flops_tok = train_flops_per_token(cfg, cfg.r_mean if cfg.recurrent else 0,
                                      min(cfg.k_bptt, cfg.r_mean) if cfg.recurrent else 0, rc.T)
    say(f"{rc.preset}: {sum(p.numel() for p in model.parameters()):,} params · {world} rangos · reparto "
        f"{[round(x / 1e6) for x in loads]}M · {rc.steps} pasos de {rc.batch * rc.T:,} tokens · "
        f"{flops_tok / 1e9:.2f} GFLOP/token · modelo {rc.precision}, maestros fp32")

    def save(tag_step):
        if not rc.ckpt_dir:
            return
        d = Path(rc.ckpt_dir)
        if rank == 0:
            d.mkdir(parents=True, exist_ok=True)
        if world > 1:
            dist.barrier()
        torch.save(dict(masters={n: m.detach().cpu() for n, m in masters.items()}, opt=opt.state_dict(), t=opt.t),
                   d / f"opt_r{rank}.tmp")
        if rank == 0:
            torch.save(dict(step=tag_step, data=data.state(), scaler=scaler.state_dict() if scaler else None,
                            extra=dict(history=history), owner=owner, world=world), d / "model.tmp")
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
            if on_save is not None:
                on_save()

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
        S = scaler.scale if scaler else 1.0
        tot = 0.0
        for i in range(n_micro):
            xa, ya = data.batch(rc.micro * world)  # todos los rangos avanzan el mismo cursor
            x = xa[rank * rc.micro:(rank + 1) * rc.micro].to(device, non_blocking=True)
            y = ya[rank * rc.micro:(rank + 1) * rc.micro].to(device, non_blocking=True)
            logits = model(x, r=r, k=k)
            loss = F.cross_entropy(logits.float().flatten(0, 1), y.flatten()) / n_micro
            (loss * S).backward()
            tot += float(loss.detach())
        # reducir cada gradiente (fp32) hacia su dueño y liberar el resto
        sq = torch.zeros(1, device=device, dtype=torch.float64)
        for n, p in named:
            g = p.grad.float() if p.grad is not None else torch.zeros(p.shape, device=device)
            p.grad = None
            if world > 1:
                dist.reduce(g, dst=owner[n], op=dist.ReduceOp.SUM)
            if owner[n] == rank:
                g.div_(world)
                masters[n].grad = g
                sq += (g.double() ** 2).sum()
            else:
                del g
        if world > 1:
            dist.all_reduce(sq)
        norm_scaled = float(sq.sqrt())
        ok = math.isfinite(norm_scaled)
        gn = norm_scaled / S
        if scaler:
            if not ok:
                scaler.scale /= 2.0
                scaler.good = 0
                scaler.skipped += 1
            else:
                scaler.good += 1
                if scaler.good % scaler.interval == 0:
                    scaler.scale = min(scaler.scale * 2.0, scaler.max)
        if ok:
            coef = (min(1.0, rc.clip / (gn + 1e-6)) if rc.clip else 1.0) / S
            for m in masters.values():
                m.grad.mul_(coef)
            opt.step(lr_mult(step, rc))
            publish()
        for m in masters.values():
            m.grad = None
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
                f"saltados {row['skipped']}" + (f" mem {row['mem_gb']:.1f} GB" if use_cuda else ""))
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


def train_ddp_run(kw):
    """Punto de entrada para experiments.run_grid (un proceso por GPU, world = 1)."""
    kw = dict(kw)
    ns_dtype = kw.pop("ns_dtype", "fp16")
    return train_ddp(LMRun(**kw), ns_dtype=ns_dtype, log=lambda s: print(s, flush=True))
