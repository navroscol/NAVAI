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
