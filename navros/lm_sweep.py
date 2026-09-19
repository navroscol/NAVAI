"""Muon contra AdamW en un modelo de lenguaje a nivel de bytes, comparando mejor contra mejor.

Protocolo (por ancho y por horizonte de pasos):
  1. AdamW: barrido de LR con semilla 0; se elige por pérdida de validación final.
  2. Muon (matrices) + Adam (emb y ganancias con el LR óptimo del paso 1): barrido del
     LR de Muon con semilla 0; se elige igual.
  3. Los dos ganadores con semillas 1 y 2 (más la 0 ya hecha); se reporta la pérdida de
     PRUEBA (documentos que nunca se usaron para elegir nada).
Todos ven los mismos lotes por semilla. También se mide el tiempo por paso.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import NavrosConfig
from .data import NpzCorpus
from .pt.model import Navros
from .pt.optim import AdamW, Muon, clip_grad_norm


def lm_config(width, depth):
    return NavrosConfig(vocab=256, d=width, n_heads=max(1, width // 64), n_pre=depth, n_core=0, causal=True)


@torch.no_grad()
def eval_loss(model, batches, device):
    model.eval()
    tot = 0.0
    for x, y in batches:
        x, y = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
        tot += float(F.cross_entropy(model(x).flatten(0, 1), y.flatten()))
    model.train()
    return tot / len(batches)


def train_lm(kw):
    width, depth, T, B = kw["width"], kw.get("depth", 4), kw.get("T", 256), kw.get("batch", 32)
    steps, warmup, seed = kw["steps"], kw.get("warmup", max(1, kw["steps"] // 10)), kw["seed"]
    device = torch.device(kw.get("device", "cpu"))
    corpus = NpzCorpus(kw["corpus"])
    torch.manual_seed(seed)
    cfg = lm_config(width, depth)
    model = Navros(cfg).to(device)
    named = list(model.named_parameters())
    if kw["opt"] == "muon":
        opt = Muon(named, lr_muon=kw["lr"], lr_adam=kw["lr_adam"])
    else:
        opt = AdamW(named, lr=kw["lr"])
    val = corpus.fixed_batches("val", kw.get("n_val", 16), B, T, seed=777)
    test = corpus.fixed_batches("test", kw.get("n_test", 32), B, T, seed=888)
    rng = np.random.default_rng(10_000 + seed)
    eval_every = kw.get("eval_every", max(1, steps // 20))
    curve, train_curve, t_steps = [], [], []
    run = 0.0
    for step in range(steps):
        x, y = corpus.batch(rng, B, T)
        x, y = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm(model.parameters(), 1.0)
        lr_mult = (step + 1) / warmup if step < warmup else \
            0.1 + 0.45 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup)))
        opt.step(lr_mult)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_steps.append(time.time() - t0)
        lv = float(loss.detach())
        if not math.isfinite(lv):
            return dict(kw=kw, diverged=True, step=step, params=cfg.n_params())
        run += lv
        if (step + 1) % eval_every == 0:
            curve.append((step + 1, eval_loss(model, val, device)))
            train_curve.append((step + 1, run / eval_every))
            run = 0.0
    return dict(kw=kw, diverged=False, params=cfg.n_params(), curve=curve, train_curve=train_curve,
                val=curve[-1][1], test=eval_loss(model, test, device),
                sec_per_step=float(np.median(t_steps[5:])) if len(t_steps) > 5 else float(np.mean(t_steps)))


def lm_brief(res):
    if res.get("diverged"):
        return f"DIVERGE en paso {res['step']}"
    return f"val {res['val']:.4f} test {res['test']:.4f} ({1000*res['sec_per_step']:.0f} ms/paso)"


def optimizer_study(out_dir, corpus, widths=(128, 256, 512), horizons=(250, 1000),
                    adam_lrs=(5e-4, 1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2),
                    muon_lrs=(0.005, 0.01, 0.02, 0.04, 0.08), seeds=(0, 1, 2), per_gpu=2, base=None):
    from .experiments import _ms, run_grid
    out_dir = Path(out_dir)
    base = dict(corpus=str(corpus)) | (base or {})
    g = dict(fn="navros.lm_sweep:train_lm", brief="navros.lm_sweep:lm_brief")
    name = lambda o, w, h, lr, s: f"{o}_w{w}_h{h}_lr{lr}_s{s}"
    ok = lambda r: r is not None and not r.get("diverged")

    # 1. AdamW
    runs = {name("adamw", w, h, lr, 0): base | dict(opt="adamw", width=w, steps=h, lr=lr, seed=0)
            for w in widths for h in horizons for lr in adam_lrs}
    res = run_grid(runs, out_dir, per_gpu, **g)
    best_adam = {}
    for w in widths:
        for h in horizons:
            c = {lr: res.get(name("adamw", w, h, lr, 0)) for lr in adam_lrs}
            c = {lr: r for lr, r in c.items() if ok(r)}
            best_adam[(w, h)] = min(c, key=lambda lr: c[lr]["val"])
    # 2. Muon
    runs = {name("muon", w, h, lr, 0): base | dict(opt="muon", width=w, steps=h, lr=lr, lr_adam=best_adam[(w, h)], seed=0)
            for w in widths for h in horizons for lr in muon_lrs}
    res |= run_grid(runs, out_dir, per_gpu, **g)
    best_muon = {}
    for w in widths:
        for h in horizons:
            c = {lr: res.get(name("muon", w, h, lr, 0)) for lr in muon_lrs}
            c = {lr: r for lr, r in c.items() if ok(r)}
            best_muon[(w, h)] = min(c, key=lambda lr: c[lr]["val"])
    # 3. semillas
    runs = {}
    for w in widths:
        for h in horizons:
            for s in seeds:
                runs[name("adamw", w, h, best_adam[(w, h)], s)] = base | dict(opt="adamw", width=w, steps=h, lr=best_adam[(w, h)], seed=s)
                runs[name("muon", w, h, best_muon[(w, h)], s)] = base | dict(opt="muon", width=w, steps=h, lr=best_muon[(w, h)],
                                                                             lr_adam=best_adam[(w, h)], seed=s)
    res |= run_grid(runs, out_dir, per_gpu, **g)

    summary = {}
    for w in widths:
        for h in horizons:
            A = [res[name("adamw", w, h, best_adam[(w, h)], s)] for s in seeds]
            M = [res[name("muon", w, h, best_muon[(w, h)], s)] for s in seeds]
            # paso en el que Muon (media de semillas) alcanza la pérdida de validación final de AdamW
            ca = np.mean([[v for _, v in r["curve"]] for r in A], axis=0)
            cm = np.mean([[v for _, v in r["curve"]] for r in M], axis=0)
            stp = [s for s, _ in A[0]["curve"]]
            hit = next((stp[i] for i, v in enumerate(cm) if v <= ca[-1]), None)
            summary[f"w{w}_h{h}"] = dict(
                width=w, steps=h, params=A[0]["params"],
                adamw=dict(lr=best_adam[(w, h)], test=_ms([r["test"] for r in A]), sec_per_step=_ms([r["sec_per_step"] for r in A])),
                muon=dict(lr=best_muon[(w, h)], lr_adam=best_adam[(w, h)], test=_ms([r["test"] for r in M]),
                          sec_per_step=_ms([r["sec_per_step"] for r in M])),
                muon_step_matching_adamw_final=hit,
                curves=dict(steps=stp, adamw=ca.tolist(), muon=cm.tolist()),
                sweep=dict(adamw={lr: (res[name("adamw", w, h, lr, 0)] or {}).get("val") for lr in adam_lrs},
                           muon={lr: (res[name("muon", w, h, lr, 0)] or {}).get("val") for lr in muon_lrs}))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    md = optimizer_markdown(summary)
    (out_dir / "summary.md").write_text(md)
    print(md, flush=True)
    return summary


def optimizer_markdown(summary):
    L = ["## Muon contra AdamW (LM de bytes; pérdida de prueba, media ± desv. típica, 3 semillas)", "",
         "| ancho | params | pasos | AdamW (LR) | Muon (LR) | Δ | Muon iguala el final de AdamW en el paso | s/paso AdamW | s/paso Muon |",
         "|---|---|---|---|---|---|---|---|---|"]
    for v in summary.values():
        a, m = v["adamw"], v["muon"]
        L.append(f"| {v['width']} | {v['params']:,} | {v['steps']} | {a['test']['mean']:.4f} ± {a['test']['std']:.4f} ({a['lr']}) "
                 f"| {m['test']['mean']:.4f} ± {m['test']['std']:.4f} ({m['lr']}) | {m['test']['mean'] - a['test']['mean']:+.4f} "
                 f"| {v['muon_step_matching_adamw_final'] or '—'} | {1000*a['sec_per_step']['mean']:.0f} ms "
                 f"| {1000*m['sec_per_step']['mean']:.0f} ms |")
    return "\n".join(L) + "\n"
