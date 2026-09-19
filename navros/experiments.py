"""Estudios completos que se ejecutan en Kaggle. Cada uno decide todo lo que puede con
validación en distribución y deja la prueba de longitud intacta hasta el final.

reasoner_study (por tarea):
  A. Barrido de LR (semilla 0) para bucle y fijo. Se elige el mejor de cada uno por
     exactitud de secuencia en validación a longitud L (nunca con longitudes de prueba).
  B. El mejor LR de cada arquitectura con 3 semillas.
  C. Ablación: bucle entrenado con r fijo (mismo LR elegido para el bucle), 3 semillas.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import time
import traceback
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------- rejilla GPU
def _resolve(path):
    import importlib
    mod, fn = path.split(":")
    return getattr(importlib.import_module(mod), fn)


def _worker(dev, q, out_dir, fn_path, summary_fn_path):
    import torch
    torch.set_num_threads(1)
    fn = _resolve(fn_path)
    brief = _resolve(summary_fn_path)
    while True:
        item = q.get()
        if item is None:
            return
        name, kw = item
        path = Path(out_dir) / f"{name}.json"
        if path.exists():
            continue
        try:
            t = time.time()
            res = fn(kw | dict(device=dev, tag=name))
            path.write_text(json.dumps(res, indent=1, default=float))
            print(f"[{dev}] {name}: {brief(res)}  ({time.time()-t:.0f}s)", flush=True)
        except Exception:
            (Path(out_dir) / f"{name}.error.txt").write_text(traceback.format_exc())
            print(f"[{dev}] {name}: ERROR\n{traceback.format_exc()}", flush=True)


def run_grid(runs: dict, out_dir, per_gpu=2, fn="navros.experiments:reasoner_run",
             brief="navros.experiments:reasoner_brief"):
    """Ejecuta fn(kwargs) para cada corrida, repartidas entre GPUs (per_gpu procesos por GPU).
    Las corridas ya terminadas (JSON presente) no se repiten: se puede reanudar."""
    import torch
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    n = torch.cuda.device_count()
    devs = [f"cuda:{i}" for _ in range(per_gpu) for i in range(n)] or ["cpu"]
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    for item in runs.items():
        q.put(item)
    for _ in devs:
        q.put(None)
    procs = [ctx.Process(target=_worker, args=(d, q, str(out_dir), fn, brief)) for d in devs]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    out = {}
    for name in runs:
        f = Path(out_dir) / f"{name}.json"
        if f.exists():
            out[name] = json.loads(f.read_text())
    return out


def reasoner_run(kw):
    from navros.reasoner import RunCfg, train
    return train(RunCfg(**kw), verbose=False)


def reasoner_brief(res):
    v = res["val"]["same_compute"]
    return f"val seq {v['seq']:.3f} pos {v['pos']:.3f}"


# ------------------------------------------------------------------ razonador
def _val_key(res):
    v = res["val"]["same_compute"]
    return (v["seq"], v["pos"])


def reasoner_study(task, out_dir, L=8, steps=3000, lrs=(0.005, 0.01, 0.02, 0.04), seeds=(0, 1, 2),
                   adam_ratio=0.15, per_gpu=2, base=None):
    base = dict(task=task, L=L, steps=steps) | (base or {})
    out_dir = Path(out_dir) / task
    log = {}

    def lr_kw(lr):
        return dict(lr_muon=lr, lr_adam=adam_ratio * lr)

    # A. barrido de LR, semilla 0
    runsA = {f"{a}_lr{lr}_s0": base | dict(arch=a, seed=0) | lr_kw(lr) for a in ("bucle", "fijo") for lr in lrs}
    resA = run_grid(runsA, out_dir, per_gpu)
    best = {}
    for a in ("bucle", "fijo"):
        cands = {lr: resA[f"{a}_lr{lr}_s0"] for lr in lrs if f"{a}_lr{lr}_s0" in resA}
        best[a] = max(cands, key=lambda lr: _val_key(cands[lr]))
        log[f"sweep_{a}"] = {lr: dict(val=_val_key(r)) for lr, r in cands.items()}
    log["best_lr"] = best

    # B + C. mejores LR con 3 semillas, y ablación r fijo
    runsB = {f"{a}_lr{best[a]}_s{s}": base | dict(arch=a, seed=s) | lr_kw(best[a])
             for a in ("bucle", "fijo") for s in seeds}
    runsB |= {f"bucle-rfijo_lr{best['bucle']}_s{s}": base | dict(arch="bucle", r_random=False, seed=s) | lr_kw(best["bucle"])
              for s in seeds}
    resB = run_grid(runsB, out_dir, per_gpu)
    groups = {"bucle": [], "fijo": [], "bucle-rfijo": []}
    for name, r in resB.items():
        groups[name.split("_lr")[0]].append(r)
    summary = summarize(groups, L)
    summary["selection"] = log
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    (out_dir / "summary.md").write_text(to_markdown(summary, task))
    print(to_markdown(summary, task), flush=True)
    return summary


def width_study(task, out_dir, widths=(64, 128, 256), seeds=(0, 1, 2), lr=0.01, adam_ratio=0.15, per_gpu=2, base=None):
    """Tensión ancho–generalización: el bucle a varios anchos, mismo protocolo y LR."""
    base = dict(task=task, arch="bucle", lr_muon=lr, lr_adam=adam_ratio * lr) | (base or {})
    out_dir = Path(out_dir) / f"{task}_ancho"
    runs = {f"bucle-d{w}_lr{lr}_s{s}": base | dict(d=w, heads=max(1, w // 32), seed=s) for w in widths for s in seeds}
    res = run_grid(runs, out_dir, per_gpu)
    groups = {f"bucle-d{w}": [r for n, r in res.items() if n.startswith(f"bucle-d{w}_")] for w in widths}
    summary = summarize(groups, base.get("L", 8))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    L = [f"## {task}: barrido de ancho del bucle (secuencia / posición a r̄, media de {len(seeds)} semillas)", "",
         "| ancho | params | " + " | ".join(f"L={m}" for m in next(iter(summary.values()))["test"]) + " |",
         "|---|---|" + "---|" * len(next(iter(summary.values()))["test"])]
    for g, v in summary.items():
        cells = [f"{100*r['same_compute']['seq']['mean']:.1f} / {100*r['same_compute']['pos']['mean']:.1f}" for r in v["test"].values()]
        L.append(f"| {g} | {v['params']:,} | " + " | ".join(cells) + " |")
    md = "\n".join(L) + "\n"
    (out_dir / "summary.md").write_text(md)
    print(md, flush=True)
    return summary


def _ms(xs):
    xs = np.asarray(xs, dtype=float)
    return dict(mean=float(xs.mean()), std=float(xs.std(ddof=1)) if len(xs) > 1 else 0.0, n=len(xs), vals=xs.tolist())


def summarize(groups, L):
    out = {}
    for g, runs in groups.items():
        if not runs:
            continue
        t0 = runs[0]["test"]
        rows = {}
        for key in sorted(t0, key=int):
            row = {}
            for mode in ("same_compute", "scaled", "adaptive"):
                if mode in t0[key]:
                    row[mode] = dict(seq=_ms([r["test"][key][mode]["seq"] for r in runs]),
                                     pos=_ms([r["test"][key][mode]["pos"] for r in runs]))
                    if mode == "scaled":
                        row[mode]["r"] = t0[key][mode]["r"]
                    if mode == "adaptive":
                        row[mode]["mean_r"] = _ms([r["test"][key][mode]["mean_r"] for r in runs])
            if "curve_seq" in t0[key]:
                row["curve_seq"] = np.mean([r["test"][key]["curve_seq"] for r in runs], axis=0).tolist()
                row["curve_pos"] = np.mean([r["test"][key]["curve_pos"] for r in runs], axis=0).tolist()
            rows[key] = row
        out[g] = dict(params=runs[0]["params"], train_units=_ms([r["train_units"] for r in runs]),
                      t_train=_ms([r["t_train"] for r in runs]), lr=runs[0]["run"]["lr_muon"],
                      val_seq=_ms([r["val"]["same_compute"]["seq"] for r in runs]), test=rows)
    return out


def to_markdown(s, task):
    def f(x):
        return f"{100*x['mean']:.1f} ± {100*x['std']:.1f}"
    lines = [f"## {task}: bucle contra referencia fija (media ± desv. típica, {s['bucle']['val_seq']['n']} semillas)", ""]
    g = [k for k in ("bucle", "fijo", "bucle-rfijo") if k in s]
    lines.append("| grupo | params | LR Muon | cómputo entreno (rel. fijo) | val seq (L) |")
    lines.append("|---|---|---|---|---|")
    ref = s["fijo"]["train_units"]["mean"] if "fijo" in s else 1.0
    for k in g:
        lines.append(f"| {k} | {s[k]['params']:,} | {s[k]['lr']} | {s[k]['train_units']['mean']/ref:.2f}× | {f(s[k]['val_seq'])} |")
    lines.append("")
    lens = list(s["bucle"]["test"].keys())
    lines.append("| longitud | métrica | " + " | ".join(
        ["bucle (r̄, mismo cómputo)", "bucle (r ∝ longitud)", "bucle (adaptativo)", "fijo", "bucle r fijo (r̄)"]) + " |")
    lines.append("|---|---|" + "---|" * 5)
    for m in lens:
        for met in ("seq", "pos"):
            b = s["bucle"]["test"][m]
            cells = [f(b["same_compute"][met]), f(b["scaled"][met]) + f" (r={b['scaled']['r']})",
                     f(b["adaptive"][met]) + f" (r̄={b['adaptive']['mean_r']['mean']:.1f})",
                     f(s["fijo"]["test"][m]["same_compute"][met]) if "fijo" in s else "—",
                     f(s["bucle-rfijo"]["test"][m]["same_compute"][met]) if "bucle-rfijo" in s else "—"]
            lines.append(f"| {m} | {'secuencia' if met == 'seq' else 'posición'} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ LM pequeño
def lm_study(out_dir, data_dirs, presets=("rec-s", "fix20-s", "fix8-s"), sweep_lrs=(0.01, 0.02, 0.04),
             sweep_tokens=20_000_000, full_tokens=200_000_000, adam_ratio=0.15, base=None, per_gpu=1,
             ckpt_root="/kaggle/working/ckpt"):
    """¿Ayuda la recurrencia a un LM a igualdad de cómputo?

    1. Barrido corto de LR (sweep_tokens) para cada arquitectura; se elige por pérdida de
       validación media (es+en) al final del barrido.
    2. Corrida completa (full_tokens) de cada arquitectura con su mejor LR.
    rec-s: 2+4×r̄+2 capas (r̄=4, pesos del núcleo compartidos); fix20-s: 20 capas distintas
    (mismo cómputo de forward a r̄); fix8-s: 8 capas (mismos parámetros únicos).
    """
    out_dir = Path(out_dir)
    base = dict(data_dirs=list(data_dirs), T=1024, batch=64, micro=8, precision="fp16", eval_seq=64,
                muon_buf="bf16") | (base or {})
    g = dict(fn="navros.lm_ddp:train_ddp_run", brief="navros.lm:lm_brief")  # mismo entrenador que el 1B
    lr_kw = lambda lr: dict(lr_muon=lr, lr_adam=adam_ratio * lr)
    runs = {f"{p}_lr{lr}_barrido": base | dict(preset=p, tokens=sweep_tokens, warmup=50, eval_every=10**9,
                                                decay_frac=0.2, tag=f"barrido-{lr}") | lr_kw(lr)
            for p in presets for lr in sweep_lrs}
    res = run_grid(runs, out_dir, per_gpu, **g)
    best, sweep = {}, {}
    for p in presets:
        c = {lr: res.get(f"{p}_lr{lr}_barrido") for lr in sweep_lrs}
        c = {lr: np.mean(list(r["val"].values())) for lr, r in c.items() if r and r.get("finished")}
        sweep[p] = c
        best[p] = min(c, key=c.get)
    runs = {f"{p}_lr{best[p]}_completo": base | dict(preset=p, tokens=full_tokens, warmup=200, eval_every=500,
                                                     ckpt_dir=f"{ckpt_root}/{p}", tag="completo") | lr_kw(best[p])
            for p in presets}
    res = run_grid(runs, out_dir, per_gpu, **g)
    summary = dict(sweep=sweep, best_lr=best, runs={})
    for name, r in res.items():
        p = name.split("_lr")[0]
        summary["runs"][p] = dict(params=r["params"], flops_per_token=r["flops_per_token"], finished=r["finished"],
                                  val=r.get("val"), test=r.get("test"), recurrence=r.get("recurrence"),
                                  seconds=r["seconds"], tok_s=np.median([h["tok_s"] for h in r["history"] if "tok_s" in h]))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    md = lm_markdown(summary)
    (out_dir / "summary.md").write_text(md)
    print(md, flush=True)
    return summary


def lm_markdown(s):
    L = ["## LM pequeño: recurrente contra pilas fijas (pérdida de prueba, nats/token)", "",
         "| modelo | params | GFLOP/token entreno | LR Muon | test es | test en | tok/s |", "|---|---|---|---|---|---|---|"]
    for p, r in s["runs"].items():
        t = r["test"] or {}
        L.append(f"| {p} | {r['params']:,} | {r['flops_per_token']/1e9:.2f} | {s['best_lr'][p]} | "
                 f"{t.get('es', float('nan')):.4f} | {t.get('en', float('nan')):.4f} | {r['tok_s']/1e3:.1f}K |")
    rec = s["runs"].get("rec-s", {}).get("recurrence")
    if rec:
        L += ["", "Pérdida de validación del recurrente según r (r̄ de entrenamiento = 4):", "",
              "| idioma | " + " | ".join(f"r={k}" for k in next(iter(rec.values()))["curve"]) + " | adaptativo |",
              "|---|" + "---|" * (len(next(iter(rec.values()))["curve"]) + 1)]
        for lang, d in rec.items():
            L.append(f"| {lang} | " + " | ".join(f"{v:.4f}" for v in d["curve"].values())
                     + f" | {d['adaptive']['loss']:.4f} (r̄={d['adaptive']['mean_r']:.1f}) |")
    return "\n".join(L) + "\n"
