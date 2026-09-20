"""Agrega los JSON de resultados/<plan>/ en tablas Markdown: media ± desv. típica sobre semillas.

    python resumen.py base extras
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent


def cargar(planes):
    filas = []
    for p in planes:
        for f in sorted((AQUI / "resultados" / p).glob("*.json")):
            filas.append(json.loads(f.read_text()))
    return filas


def tabla(filas, tarea, metrica="exacta"):
    grupos = defaultdict(list)
    for r in filas:
        if r["tarea"] == tarea:
            grupos[(r["modelo"], r["chi"])].append(r)
    if not grupos:
        return ""
    Ls = sorted({L for r in filas if r["tarea"] == tarea for L in r["eval"]}, key=int)
    out = [f"### {tarea} — exactitud de secuencia completa (%), media ± desv. típica, {max(len(v) for v in grupos.values())} semillas\n",
           "| modelo | χ / d | parámetros | " + " | ".join(f"L={L}" + (" (entreno)" if L == str(filas[0]["L_train"]) else "") for L in Ls) + " | s/corrida |",
           "|---|---|---|" + "---|" * len(Ls) + "---|"]
    for (m, chi), rs in sorted(grupos.items()):
        celdas = []
        for L in Ls:
            v = np.array([r["eval"][L][metrica] for r in rs]) * 100
            celdas.append(f"{v.mean():.1f} ± {v.std():.1f}")
        extra = ""
        if any("tasa_disparo" in h for h in rs[0]["historia"]):
            t = np.mean([r["historia"][-1].get("tasa_disparo", 0) for r in rs]) * 100
            extra = f" (disparo {t:.0f} %)"
        out.append(f"| {m}{extra} | {chi} | {rs[0]['parametros']:,} | " + " | ".join(celdas) + f" | {np.mean([r['segundos'] for r in rs]):.0f} |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    planes = sys.argv[1:] or ["base"]
    filas = cargar(planes)
    print(f"{len(filas)} corridas de {planes}\n")
    for tarea in ["paso-1", "paso-2", "paso-3", "parada"]:
        t = tabla(filas, tarea)
        if t:
            print(t)
    t = tabla(filas, "paso-2", "bits")
    if t:
        print(t.replace("exactitud de secuencia completa", "exactitud por bit"))
