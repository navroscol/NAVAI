"""Banco de superposición en Modal: cada corrida es un contenedor de CPU; los resultados van al
Volume `navros-superposicion` (/res) y se descargan al terminar.

    modal run investigacion/superposicion/modal_superposicion.py --plan base            # CPU, ~60 corridas
    modal run investigacion/superposicion/modal_superposicion.py --plan extras          # CPU
    modal run investigacion/superposicion/modal_superposicion.py --plan grande --gpu    # T4, χ=32, L=16→48

Los modelos son diminutos (χ=16..32, Transformer d≤128), así que una CPU por corrida basta para
`base` y `extras`; la gracia de Modal es el abanico: todas las corridas a la vez. El plan `grande`
(6000 pasos, longitudes hasta 48) va en T4, una GPU por corrida.
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

AQUI = Path(__file__).resolve().parent
REMOTO = "/root/superposicion"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.10.0", "numpy")
    .add_local_dir(AQUI, REMOTO, ignore=["**/__pycache__/**", "resultados/**"])
)
app = modal.App("navros-superposicion", image=image)
vol = modal.Volume.from_name("navros-superposicion", create_if_missing=True)


def _corrida(cfg: dict, plan: str, pasos: int, hilos: int) -> dict:
    import sys
    sys.path.insert(0, REMOTO)
    import experimentos
    cfg = dict(cfg)
    cfg.setdefault("pasos", pasos)
    ruta = Path("/res") / plan / (experimentos.etiqueta(cfg) + ".json")
    if ruta.exists():
        return json.loads(ruta.read_text())
    res = experimentos.correr(**cfg, hilos=hilos, log=lambda s: print(s, flush=True))
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(res, indent=1))
    vol.commit()
    return res


@app.function(cpu=2, memory=4096, timeout=2 * 3600, volumes={"/res": vol})
def corrida_cpu(cfg: dict, plan: str, pasos: int) -> dict:
    return _corrida(cfg, plan, pasos, hilos=2)


@app.function(gpu="T4", cpu=4, memory=8192, timeout=3 * 3600, volumes={"/res": vol})
def corrida_gpu(cfg: dict, plan: str, pasos: int) -> dict:
    return _corrida(cfg, plan, pasos, hilos=4)


@app.local_entrypoint()
def main(plan: str = "base", pasos: int = 3000, gpu: bool = False):
    import sys
    sys.path.insert(0, str(AQUI))
    import experimentos
    cfgs = experimentos.plan(plan)
    fn = corrida_gpu if gpu else corrida_cpu
    print(f"plan {plan}: {len(cfgs)} corridas en Modal ({'T4' if gpu else 'CPU'})")
    carpeta = AQUI / "resultados" / plan
    carpeta.mkdir(parents=True, exist_ok=True)
    for res in fn.starmap([(c, plan, pasos) for c in cfgs]):
        cfg = {k: res[k] for k in ("modelo", "tarea", "chi", "semilla")}
        (carpeta / (experimentos.etiqueta(cfg) + ".json")).write_text(json.dumps(res, indent=1))
        print("listo", experimentos.etiqueta(cfg), {k: round(v["exacta"], 3) for k, v in res["eval"].items()})
    print("resultados en", carpeta)
