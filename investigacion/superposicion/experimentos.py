"""Entrenamiento y evaluación del banco de superposición.

    python investigacion/superposicion/experimentos.py --rapido          # una corrida de prueba
    python investigacion/superposicion/experimentos.py --plan base       # el plan completo, en paralelo (CPU)

Protocolo (el mismo que en navros/): se entrena con longitudes mezcladas 3..L_train, se elige nada
con la prueba fuera de distribución; las longitudes largas se miden UNA vez al final.
Cada corrida escribe un JSON en resultados/<plan>/ con la curva y las exactitudes por longitud.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))

L_TRAIN = 12
L_TEST = [12, 16, 24, 32]


def correr(modelo: str, tarea: str, chi: int = 16, semilla: int = 0, pasos: int = 3000, lote: int = 128,
           lr: float | None = None, L_train: int = L_TRAIN, L_test=tuple(L_TEST), n_eval: int = 512,
           hilos: int = 1, log=print) -> dict:
    import torch
    import collatz
    import modelos
    torch.set_num_threads(hilos)
    torch.manual_seed(semilla)
    disp = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(semilla)
    secuencia = tarea.startswith("paso-")
    m = modelos.construir(modelo, chi, secuencia).to(disp)
    if lr is None:
        lr = 1e-3 if modelo.startswith("tf-") else 1e-2
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.98))
    calendario = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * min(s, pasos) / pasos)))
    t0, historia = time.time(), []
    m.train()
    for paso in range(pasos):
        L = int(rng.integers(3, L_train + 1))
        x, y = collatz.lote(tarea, rng, L, lote)
        x, y = torch.from_numpy(x).to(disp), torch.from_numpy(y).to(disp)
        logp = m(x)
        perdida = -logp.gather(-1, y.unsqueeze(-1)).mean()
        if hasattr(m, "parciales"):  # cabezas duales: cada cabeza debe ser competente por sí sola
            for lp in m.parciales:
                perdida = perdida + 0.5 * (-lp.gather(-1, y.unsqueeze(-1)).mean())
        opt.zero_grad(set_to_none=True)
        perdida.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        calendario.step()
        if paso % 100 == 0 or paso == pasos - 1:
            with torch.no_grad():
                pred = logp.argmax(-1)
                ex = (pred == y).all(-1).float().mean().item() if secuencia else (pred == y).float().mean().item()
            h = dict(paso=paso, L=L, perdida=round(perdida.item(), 4), exacta=round(ex, 4), t=round(time.time() - t0, 1))
            if hasattr(m, "tasa"):
                h["tasa_disparo"] = round(float(m.tasa.detach()), 4)
            historia.append(h)
            log(f"  {modelo} {tarea} χ={chi} s={semilla} " + " ".join(f"{k}={v}" for k, v in h.items()))
    m.eval()
    evals = {}
    rng_eval = np.random.default_rng(10_000 + semilla)
    with torch.no_grad():
        for L in L_test:
            x, y = collatz.lote(tarea, rng_eval, L, n_eval)
            x, y = torch.from_numpy(x).to(disp), torch.from_numpy(y).to(disp)
            pred = m(x).argmax(-1)
            if secuencia:
                evals[str(L)] = dict(exacta=(pred == y).all(-1).float().mean().item(),
                                     bits=(pred == y).float().mean().item())
            else:
                evals[str(L)] = dict(exacta=(pred == y).float().mean().item())
    return dict(modelo=modelo, tarea=tarea, chi=chi, semilla=semilla, pasos=pasos, lote=lote, lr=lr,
                L_train=L_train, parametros=modelos.n_parametros(m), segundos=round(time.time() - t0, 1),
                historia=historia, eval=evals)


# ----------------------------------------------------------------------------- planes
def plan(nombre: str, semillas=(0, 1, 2)) -> list[dict]:
    cfgs = []
    if nombre in ("base", "todo"):
        for tarea in ["paso-1", "paso-2", "paso-3"]:
            for modelo in ["mps-real", "mps-complejo", "mps-unitario", "mps-fourier", "tf-rope", "tf-abaco"]:
                for s in semillas:
                    cfgs.append(dict(modelo=modelo, tarea=tarea, chi=16, semilla=s))
        for modelo in ["mps-complejo", "tf-rope"]:
            for s in semillas:
                cfgs.append(dict(modelo=modelo, tarea="parada", chi=16, semilla=s))
    if nombre in ("extras", "todo"):
        for modelo in ["mps-espigas", "mps-dual-critica", "mps-dual-suma"]:
            for s in semillas:
                cfgs.append(dict(modelo=modelo, tarea="paso-2", chi=16, semilla=s))
        for modelo in ["mps-complejo", "mps-fourier"]:
            for s in semillas:
                cfgs.append(dict(modelo=modelo, tarea="paso-2", chi=32, semilla=s))
    if nombre in ("extras2", "todo"):  # crítica cruzada y espigas donde χ=16 NO satura: paso-3
        for modelo in ["mps-dual-critica", "mps-dual-suma", "mps-espigas"]:
            for s in semillas:
                cfgs.append(dict(modelo=modelo, tarea="paso-3", chi=16, semilla=s))
    if nombre == "profundidad":  # ¿el 100 % en paso-3 era de la profundidad o del silencio?
        for modelo in ["mps-complejo", "mps-2capas", "mps-espigas"]:
            for s in semillas:
                cfgs.append(dict(modelo=modelo, tarea="paso-3", chi=16, semilla=s))
    if nombre == "escala":  # escala ordinal: mayor k resuelto exacto y general por cada χ
        for chi in (8, 16, 32, 64):
            for k in range(1, 7):
                for s in semillas:
                    cfgs.append(dict(modelo="mps-complejo", tarea=f"paso-{k}", chi=chi, semilla=s))
        for chi in (16, 32):  # Transformer d=64 y d=128 como referencia
            for k in range(1, 7):
                for s in semillas:
                    cfgs.append(dict(modelo="tf-rope", tarea=f"paso-{k}", chi=chi, semilla=s))
    if nombre == "escala-local":  # versión reducida para una CPU: 1 semilla, χ ≤ 32, k ≤ 5
        for chi in (8, 16, 32):
            for k in range(1, 6):
                cfgs.append(dict(modelo="mps-complejo", tarea=f"paso-{k}", chi=chi, semilla=0))
    if nombre == "grande":  # para GPU/Modal: más ancho, más largo, más pasos; L_train=16, prueba hasta 48
        for tarea in ["paso-1", "paso-2", "paso-3"]:
            for modelo in ["mps-complejo", "mps-unitario", "mps-fourier", "mps-real", "tf-rope", "tf-abaco",
                           "mps-dual-critica", "mps-dual-suma", "mps-espigas"]:
                for s in semillas:
                    cfgs.append(dict(modelo=modelo, tarea=tarea, chi=32, semilla=s, pasos=6000,
                                     L_train=16, L_test=(16, 24, 32, 48)))
        for modelo in ["mps-complejo", "mps-unitario", "tf-rope"]:
            for s in semillas:
                cfgs.append(dict(modelo=modelo, tarea="parada", chi=32, semilla=s, pasos=6000,
                                 L_train=16, L_test=(16, 24, 32, 48)))
    if not cfgs:
        raise ValueError(nombre)
    return cfgs


def etiqueta(c: dict) -> str:
    return f"{c['modelo']}_{c['tarea']}_chi{c['chi']}_s{c['semilla']}"


def _trabajo(args):
    c, carpeta = args
    ruta = Path(carpeta) / (etiqueta(c) + ".json")
    if ruta.exists():
        return json.loads(ruta.read_text())
    res = correr(**c, log=lambda s: None)  # noqa
    ruta.write_text(json.dumps(res, indent=1))
    print(f"listo {etiqueta(c)} {res['segundos']}s eval={ {k: round(v['exacta'], 3) for k, v in res['eval'].items()} }", flush=True)
    return res


def ejecutar_plan(nombre: str, carpeta: Path, procesos: int):
    from multiprocessing import Pool
    carpeta.mkdir(parents=True, exist_ok=True)
    cfgs = plan(nombre)
    print(f"plan {nombre}: {len(cfgs)} corridas, {procesos} procesos", flush=True)
    with Pool(procesos) as p:
        return list(p.imap_unordered(_trabajo, [(c, str(carpeta)) for c in cfgs]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true")
    ap.add_argument("--plan", default=None)
    ap.add_argument("--procesos", type=int, default=max(1, (os.cpu_count() or 2)))
    ap.add_argument("--modelo", default="mps-complejo")
    ap.add_argument("--tarea", default="paso-1")
    ap.add_argument("--pasos", type=int, default=300)
    a = ap.parse_args()
    if a.rapido:
        r = correr(a.modelo, a.tarea, pasos=a.pasos, hilos=os.cpu_count() or 1)
        print(json.dumps(r["eval"], indent=1), r["segundos"], "s")
    elif a.plan:
        ejecutar_plan(a.plan, AQUI / "resultados" / a.plan, a.procesos)
