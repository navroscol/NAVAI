"""Entrenamiento y evaluación del razonador: bucle (recurrente) contra referencia fija.

Reglas que este módulo impone:
  * Bucle y fijo ven exactamente los mismos lotes (RNG de datos separado del RNG de r).
  * El fijo tiene D = round(r̄) bloques distintos: mismo cómputo de forward que el bucle
    a su r medio de entrenamiento. El cómputo de entrenamiento se contabiliza aparte.
  * El learning rate se elige con validación en distribución (longitudes ≤ L), nunca con
    las longitudes de prueba.
  * Se reportan dos métricas (secuencia completa y posición) en L, 2L, 3L, 4L.
  * Para el bucle, r de evaluación se fija con reglas declaradas de antemano:
      - r = r̄              (mismo cómputo que el fijo)
      - r = r̄·m/L          (escala con la longitud)
      - adaptativo          (para cuando ‖Δh‖/‖h‖ < tol; tope r_eval_max)
    y además se guarda la curva completa por r, solo para transparencia.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from .config import NavrosConfig
from .oracle.model import sample_r
from .pt.model import Navros, weighted_xent
from .pt.optim import AdamW, Muon, clip_grad_norm
from .tasks import expr, suma


@dataclass
class RunCfg:
    task: str = "suma"
    L: int = 8
    arch: str = "bucle"            # bucle | fijo
    d: int = 128
    heads: int = 4
    n_core: int = 1
    r_mean: float = 8.0
    r_sigma: float = 0.5
    r_max: int = 24
    r_random: bool = True          # False = ablación: r fijo = round(r_mean) en entrenamiento
    k_bptt: int = 4
    blocks: int = 0                # fijo: 0 → round(r_mean)
    rope: bool = True
    rope_range: int = 0            # >0: posiciones RoPE aleatorizadas, subconjunto ordenado de [0, rope_range)
    abacus: int = 0                # 0 → 8L+2 en suma; en expr 16L+8 si rope=False (posición absoluta + desplazamiento)
    steps: int = 3000
    batch: int = 128
    opt: str = "muon"
    lr_muon: float = 0.02
    lr_adam: float = 3e-3
    wd: float = 0.0
    warmup: int = 100
    clip: float = 1.0
    seed: int = 0
    device: str = "cpu"
    eval_mult: tuple = (1, 2, 3, 4)
    n_eval: int = 1000
    r_eval_max: int = 64
    adaptive_tol: float = 1e-3
    pool_per_len: int = 20000
    log_every: int = 100
    tag: str = ""

    def model_cfg(self) -> NavrosConfig:
        task = TASKS[self.task]
        if self.task == "suma":
            ab = self.abacus or 8 * self.L + 2
        else:
            ab = 0 if self.rope else (self.abacus or 16 * self.L + 8)
        return NavrosConfig(vocab=task.VOCAB, d=self.d, n_heads=self.heads, n_core=self.n_core,
                            n_blocks=1 if self.arch == "bucle" else (self.blocks or round(self.r_mean)),
                            rope=self.rope, abacus=ab, r_mean=self.r_mean, r_sigma=self.r_sigma,
                            r_max=self.r_max, k_bptt=self.k_bptt)


TASKS = {"suma": suma, "expr": expr}


def build_pools(rc: RunCfg, mcfg: NavrosConfig):
    """Pools de entrenamiento, validación (≤L) y prueba (m·L), descontaminados."""
    t = TASKS[rc.task]
    rng_tr = np.random.default_rng(1000 + rc.seed)
    rng_va = np.random.default_rng(2000 + rc.seed)
    rng_te = np.random.default_rng(3000)                      # prueba idéntica para todas las corridas
    lengths = list(range(1, rc.L + 1))
    test_lens = [m * rc.L for m in rc.eval_mult]
    if rc.task == "suma":
        rr = rc.rope_range
        tr = t.build_pool(rng_tr, lengths, rc.pool_per_len, mcfg.abacus, train=True, rope_range=rr)
        va = t.build_pool(rng_va, [rc.L], rc.n_eval, mcfg.abacus, train=False, rope_range=rr)
        te = t.build_pool(rng_te, test_lens, rc.n_eval, mcfg.abacus, train=False, rope_range=rr)
    else:
        rr = rc.rope_range
        tr = t.build_pool(rng_tr, lengths, rc.pool_per_len, abacus_size=mcfg.abacus, train=True, rope_range=rr)
        va = t.build_pool(rng_va, [rc.L], rc.n_eval, abacus_size=mcfg.abacus, train=False, rope_range=rr)
        te = t.build_pool(rng_te, test_lens, rc.n_eval, abacus_size=mcfg.abacus, train=False, rope_range=rr)
    train_keys = tr.keys()
    dropped = {n: te.drop(n, train_keys) for n in test_lens} | {"val": va.drop(rc.L, train_keys)}
    return tr, va, te, dropped


def to_dev(b, device, dtype=torch.float32):
    b = dict(b, weights=b["weights"].astype(np.float32))
    return {k: torch.from_numpy(v).to(device) for k, v in b.items()}


def lr_at(step, rc: RunCfg):
    if step < rc.warmup:
        return (step + 1) / rc.warmup
    p = (step - rc.warmup) / max(1, rc.steps - rc.warmup)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p))


class Acc:
    def __init__(self):
        self.pos = self.npos = self.seq = self.nseq = 0

    def add(self, logits, b):
        pred = logits.argmax(-1)
        sup = b["weights"] > 0
        hit = (pred == b["targets"]) & sup
        self.pos += int(hit.sum())
        self.npos += int(sup.sum())
        self.seq += int((hit | ~sup).all(dim=1).sum())
        self.nseq += b["tokens"].shape[0]

    def get(self):
        return dict(pos=self.pos / max(1, self.npos), seq=self.seq / max(1, self.nseq))


@torch.no_grad()
def evaluate(model: Navros, pool, n, rc: RunCfg, device, eval_bs=250):
    """Fijo: una métrica. Bucle: métrica tras cada r ≤ r_eval_max + adaptativo."""
    model.eval()
    cfg = model.cfg
    if not cfg.recurrent:
        acc = Acc()
        for b in pool.iter_batches(n, eval_bs):
            b = to_dev(b, device)
            acc.add(model(b["tokens"], b.get("abacus"), b.get("valid"), pos=b.get("pos")), b)
        model.train()
        return dict(fixed=acc.get())
    accs = [Acc() for _ in range(rc.r_eval_max)]
    ada, used, delta_curve = Acc(), [], np.zeros(rc.r_eval_max)
    for b in pool.iter_batches(n, eval_bs):
        b = to_dev(b, device)
        x0, rope, mask = model.prelude(b["tokens"], b.get("abacus"), b.get("valid"), b.get("pos"))
        h = torch.zeros_like(x0)
        B = x0.shape[0]
        done = torch.zeros(B, dtype=torch.bool, device=device)
        it = torch.full((B,), rc.r_eval_max, device=device)
        frozen = None
        for t in range(rc.r_eval_max):
            hn = model._iter(h, x0, 0, rope, mask)
            delta = (hn - h).flatten(1).norm(dim=1) / hn.flatten(1).norm(dim=1)
            delta_curve[t] += float(delta.sum())
            h = hn
            logits = model.coda_logits(h, rope, mask)
            accs[t].add(logits, b)
            newly = (~done) & (delta < rc.adaptive_tol)
            if frozen is None:
                frozen = logits.clone()
            frozen = torch.where((~done)[:, None, None], logits, frozen)
            it = torch.where(newly, torch.full_like(it, t + 1), it)
            done |= newly
        ada.add(frozen, b)
        used.append(it.float().cpu())
    model.train()
    curve = [a.get() for a in accs]
    N = pool.size(n)
    return dict(curve=curve, adaptive=ada.get() | dict(mean_r=float(torch.cat(used).mean())),
                delta=(delta_curve / N).tolist())


def train(rc: RunCfg, verbose=True) -> dict:
    torch.manual_seed(rc.seed)
    device = torch.device(rc.device)
    mcfg = rc.model_cfg()
    t0 = time.time()
    tr, va, te, dropped = build_pools(rc, mcfg)
    t_data = time.time() - t0
    model = Navros(mcfg).to(device)
    named = list(model.named_parameters())
    if rc.opt == "muon":
        opt = Muon(named, lr_muon=rc.lr_muon, lr_adam=rc.lr_adam, wd_muon=rc.wd, wd_adam=0.0)
    else:
        opt = AdamW(named, lr=rc.lr_adam, wd=rc.wd)
    rng_data = np.random.default_rng(10 + rc.seed)
    rng_r = np.random.default_rng(20 + rc.seed)
    log, units, t_train = [], 0.0, time.time()
    run_loss = 0.0
    for step in range(rc.steps):
        b, n = tr.sample(rng_data, rc.batch)
        b = to_dev(b, device)
        if mcfg.recurrent:
            r = sample_r(rng_r, mcfg) if rc.r_random else round(rc.r_mean)
            k = min(rc.k_bptt, r)
        else:
            r = k = mcfg.n_blocks
        logits = model(b["tokens"], b.get("abacus"), b.get("valid"), r=r, k=k, pos=b.get("pos"))
        loss = weighted_xent(logits, b["targets"], b["weights"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = clip_grad_norm(model.parameters(), rc.clip)
        opt.step(lr_at(step, rc))
        T = b["tokens"].shape[1]
        units += rc.batch * T * mcfg.n_core * (r + 2 * k)     # forward 1×, backward 2× por capa-token
        run_loss += float(loss.detach())
        if (step + 1) % rc.log_every == 0:
            row = dict(step=step + 1, loss=run_loss / rc.log_every, gnorm=float(gn), t=time.time() - t_train)
            log.append(row)
            run_loss = 0.0
            if verbose:
                print(f"  [{rc.tag or rc.arch}] paso {step+1:5d}  loss {row['loss']:.4f}  |g| {row['gnorm']:.2f}  {row['t']:.0f}s", flush=True)
    t_train = time.time() - t_train

    # --- evaluación -----------------------------------------------------------
    r_bar = round(rc.r_mean)
    def summarize(ev, m):
        if "fixed" in ev:
            return dict(same_compute=ev["fixed"])
        curve = ev["curve"]
        r_scaled = min(rc.r_eval_max, max(1, round(rc.r_mean * m / rc.L)))
        return dict(same_compute=curve[r_bar - 1] | dict(r=r_bar),
                    scaled=curve[r_scaled - 1] | dict(r=r_scaled),
                    adaptive=ev["adaptive"],
                    curve_seq=[c["seq"] for c in curve], curve_pos=[c["pos"] for c in curve],
                    delta=ev["delta"])

    val = summarize(evaluate(model, va, rc.L, rc, device), rc.L)
    test = {}
    for mult in rc.eval_mult:
        m = mult * rc.L
        test[m] = summarize(evaluate(model, te, m, rc, device), m)
    res = dict(run=asdict(rc), model=mcfg.to_dict(), params=sum(p.numel() for p in model.parameters()),
               train_units=units, t_data=t_data, t_train=t_train, dropped=dropped, log=log, val=val, test=test)
    if verbose:
        print(f"  [{rc.tag or rc.arch}] params {res['params']:,}  val(L={rc.L}) seq {val['same_compute']['seq']:.3f}  "
              f"entreno {t_train:.0f}s")
        for m, v in test.items():
            s = f"    L={m:<3} mismo cómputo: seq {v['same_compute']['seq']:.3f} pos {v['same_compute']['pos']:.3f}"
            if "scaled" in v:
                s += (f" | r={v['scaled']['r']}: seq {v['scaled']['seq']:.3f} pos {v['scaled']['pos']:.3f}"
                      f" | adapt (r̄={v['adaptive']['mean_r']:.1f}): seq {v['adaptive']['seq']:.3f}")
            print(s, flush=True)
    return res


def save(res, path):
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(res, indent=1, default=float))
