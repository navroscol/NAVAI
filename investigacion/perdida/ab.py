"""A/B de técnicas para bajar la pérdida a igual número de tokens, con el entrenador REAL de navros
(navros/lm_ddp.py) y un modelo diminuto en CPU sobre un corpus local de código Python.

No mide la pérdida del 1B: mide si una técnica mueve la pérdida en la dirección esperada con el
mismo optimizador, el mismo calendario WSD y el mismo cargador de datos que usa el 1B. Sirve para
descartar y ordenar candidatos antes de gastar un workspace de 30 $ en cada uno.

    python investigacion/perdida/ab.py --bench                       # velocidad en esta CPU
    python investigacion/perdida/ab.py --variantes base,wd,lr2x,T2x  # corre y tabula

Variantes:
  base   Muon lr 0,02 (el del 1B), wd 0, WSD 20 % de decaimiento, T=256
  wd     igual + weight decay 0,1 en las matrices (Muon)          [candidato: Liu et al. 2025, Muon escalable]
  lr2x   igual con lr 0,04                                        [el barrido del 1B eligió el borde superior]
  lrhalf igual con lr 0,01                                        [por si el borde era ruido]
  T2x    igual con T=512 y la mitad de secuencias (mismos tokens) [más contexto por predicción]
  prom   base sin decaimiento + promedio de los últimos checkpoints [Hägele et al. 2024: el promedio
         sustituye al decaimiento]; se evalúa el promedio y también el último checkpoint sin promediar
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import navros.config as C
import navros.lm as lm
import navros.lm_ddp as ddp
from navros.lm import LMRun

CORPUS = os.environ.get("CORPUS_PY", "/tmp/claude-0/-home-user-NAVAI/a00da8d0-0496-505d-8e1a-4d55c8dee602/scratchpad/corpus_py")
SALIDA = Path(__file__).resolve().parent / "resultados"
CKPT = Path("/tmp/ab_ckpt")


def preset_cpu(name, vocab=32768):
    """Modelo pequeño con la misma forma que navros-1b-fix (pila fija, SwiGLU, RoPE): 4 capas, d=128."""
    if name == "cpu-s":
        return C.NavrosConfig(vocab=vocab, causal=True, rope=True, d=128, n_heads=4, ffn=384, n_pre=4, n_core=0)
    return lm.lm_preset(name, vocab)


ddp.lm_preset = preset_cpu  # el entrenador construye el modelo con esta función


def correr(nombre: str, tokens: int, T: int = 256, batch: int = 64, lr: float = 0.02, wd: float = 0.0,
           decay_frac: float = 0.2, seed: int = 0, snapshots: list | None = None, init_from: str = "",
           ckpt_every_min: float = 1e9, log=print) -> dict:
    d = CKPT / nombre
    shutil.rmtree(d, ignore_errors=True)
    rc = LMRun(preset="cpu-s", data_dirs=[CORPUS], langs={"en": 1.0}, T=T, batch=batch, micro=batch, tokens=tokens,
               lr_muon=lr, lr_adam=lr / 20, wd=wd, warmup=50, decay_frac=decay_frac, precision="fp32", device="cpu",
               seed=seed, eval_every=10 ** 9, eval_seq=128, r_eval=(1,), log_every=100, init_from=init_from,
               ckpt_dir=str(d) if snapshots is not None else "", ckpt_every_min=ckpt_every_min, tag=nombre)

    def on_save():  # guarda una copia de los maestros en cada checkpoint (para el promedio)
        so = torch.load(d / "opt_r0.pt", map_location="cpu", weights_only=False)
        snapshots.append({n: t.clone() for n, t in so["masters"].items()})

    t0 = time.time()
    res = ddp.train_ddp(rc, ns_dtype=None, log=log, on_save=on_save if snapshots is not None else None)
    return dict(nombre=nombre, val=res["val"]["en"], test=res["test"]["en"], pasos=res["step"], seg=round(time.time() - t0))


def evaluar_pesos(nombre: str, state: dict, T: int, log=print) -> dict:
    """Evalúa unos pesos con el propio entrenador: 1 paso con LR ~0 y luego su evaluación final."""
    exp = CKPT / f"{nombre}_export.pt"
    torch.save(dict(step=0, preset="cpu-s", tag=nombre, vocab=32768, state={n: t.to(torch.bfloat16) for n, t in state.items()}), exp)
    return correr(nombre + "-eval", tokens=64 * T, T=T, batch=64, lr=1e-9, init_from=str(exp), log=log)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--variantes", default="base,wd,lr2x,lrhalf,T2x,prom")
    ap.add_argument("--tokens", type=int, default=8_000_000)
    ap.add_argument("--semillas", default="0")
    ap.add_argument("--hilos", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.hilos)
    SALIDA.mkdir(exist_ok=True)
    if a.bench:
        r = correr("bench", tokens=64 * 256 * 30, log=lambda s: None)
        print(f"30 pasos de {64*256:,} tokens en {r['seg']} s → {64*256*30/r['seg']:,.0f} tok/s; val {r['val']:.3f}")
        sys.exit()
    filas = []
    for s in [int(x) for x in a.semillas.split(",")]:
        for v in a.variantes.split(","):
            t0 = time.time()
            quiet = lambda m: None
            if v == "base":
                r = correr(v, a.tokens, seed=s, log=quiet)
            elif v == "wd":
                r = correr(v, a.tokens, wd=0.1, seed=s, log=quiet)
            elif v == "lr2x":
                r = correr(v, a.tokens, lr=0.04, seed=s, log=quiet)
            elif v == "lrhalf":
                r = correr(v, a.tokens, lr=0.01, seed=s, log=quiet)
            elif v == "T2x":
                r = correr(v, a.tokens, T=512, batch=32, seed=s, log=quiet)
            elif v == "prom":
                snaps = []
                r_fin = correr("prom-sin-decaer", a.tokens, decay_frac=0.0, seed=s, snapshots=snaps, ckpt_every_min=0.25, log=quiet)
                ultimos = snaps[-max(3, len(snaps) // 3):]
                media = {n: torch.stack([sn[n] for sn in ultimos]).mean(0) for n in ultimos[0]}
                r_prom = evaluar_pesos("prom", media, T=256, log=quiet)
                r_fin["nombre"] = "prom-sin-decaer(último)"
                filas.append(dict(r_fin, semilla=s))
                r = dict(nombre=f"prom(media de {len(ultimos)} ckpt)", val=r_prom["val"], test=r_prom["test"], pasos=r_fin["pasos"], seg=r_fin["seg"])
            else:
                raise ValueError(v)
            r["semilla"] = s
            filas.append(r)
            print(f"{r['nombre']:28s} s={s} val {r['val']:.4f} test {r['test']:.4f} ({r['seg']} s)", flush=True)
            (SALIDA / "ab.jsonl").open("a").write(json.dumps(r) + "\n")
    base = {r["semilla"]: r for r in filas if r["nombre"] == "base"}
    print("\n| variante | semilla | val | test | Δ test vs base |\n|---|---|---|---|---|")
    for r in filas:
        b = base.get(r["semilla"])
        d = f"{r['test'] - b['test']:+.4f}" if b else ""
        print(f"| {r['nombre']} | {r['semilla']} | {r['val']:.4f} | {r['test']:.4f} | {d} |")
