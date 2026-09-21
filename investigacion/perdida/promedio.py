"""Promedio de checkpoints (pesos) de la fase estable: evaluar "como si hubiera decaído" sin decaer.

    python investigacion/perdida/promedio.py salida.pt export/paso_010000/pesos_bf16.pt export/paso_013500/pesos_bf16.pt ...

Toma N exports (`pesos_bf16.pt` de cloud/modal_cadena.py o de cloud/modal_pesos.py), promedia sus
tensores en fp32 y escribe un export con el mismo formato (state en bf16), listo para
`scripts/08_conversar.py --pesos` o para `LMRun(init_from=...)`. Con `--ema 0.9` hace media
exponencial (los últimos pesan más) en vez de media simple.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

ap = argparse.ArgumentParser()
ap.add_argument("salida")
ap.add_argument("exports", nargs="+")
ap.add_argument("--ema", type=float, default=0.0, help="0 = media simple; 0<β<1 = media exponencial en el orden dado")
a = ap.parse_args()

acum, pesos, meta = None, 0.0, None
for i, ruta in enumerate(a.exports):
    ex = torch.load(ruta, map_location="cpu", weights_only=False)
    meta = meta or {k: v for k, v in ex.items() if k != "state"}
    w = 1.0 if a.ema == 0 else (1 - a.ema) * a.ema ** (len(a.exports) - 1 - i)
    st = {n: t.float() * w for n, t in ex["state"].items()}
    acum = st if acum is None else {n: acum[n] + st[n] for n in acum}
    pesos += w
    print(f"  {ruta} (paso {ex.get('step')}, peso {w:.3f})")
state = {n: (t / pesos).to(torch.bfloat16) for n, t in acum.items()}
torch.save(dict(meta, state=state, promedio_de=[str(p) for p in a.exports], ema=a.ema), a.salida)
sha = hashlib.sha256(open(a.salida, "rb").read()).hexdigest()
Path(a.salida).with_suffix(".json").write_text(json.dumps(dict(meta, promedio_de=a.exports, ema=a.ema, sha256=sha), indent=1, default=str))
print(f"escrito {a.salida} ({len(a.exports)} exports, sha256 {sha[:12]}…)")
