"""Banco de superposición en Modal: cada corrida es un contenedor de CPU; los resultados van al
Volume `navros-superposicion` (/res) y se descargan al terminar.

    modal run investigacion/superposicion/modal_superposicion.py --plan base
    modal run investigacion/superposicion/modal_superposicion.py --plan extras
    modal run investigacion/superposicion/modal_superposicion.py --plan todo --pasos 6000

Los modelos son diminutos (χ=16..32, Transformer d=64), así que una CPU por corrida basta; la
gracia de Modal aquí es el abanico: ~70 corridas a la vez en vez de horas en serie.
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


@app.function(cpu=2, memory=4096, timeout=2 * 3600, volumes={"/res": vol})
def corrida(cfg: dict, plan: str, pasos: int) -> dict:
    import sys
    sys.path.insert(0, REMOTO)
    import experimentos
    ruta = Path("/res") / plan / (experimentos.etiqueta(cfg) + ".json")
    if ruta.exists():
        return json.loads(ruta.read_text())
    res = experimentos.correr(**cfg, pasos=pasos, hilos=2, log=lambda s: print(s, flush=True))
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(res, indent=1))
    vol.commit()
    return res


@app.local_entrypoint()
def main(plan: str = "base", pasos: int = 3000):
    import sys
    sys.path.insert(0, str(AQUI))
    import experimentos
    cfgs = experimentos.plan(plan)
    print(f"plan {plan}: {len(cfgs)} corridas en Modal")
    carpeta = AQUI / "resultados" / plan
    carpeta.mkdir(parents=True, exist_ok=True)
    for res in corrida.starmap([(c, plan, pasos) for c in cfgs]):
        cfg = {k: res[k] for k in ("modelo", "tarea", "chi", "semilla")}
        (carpeta / (experimentos.etiqueta(cfg) + ".json")).write_text(json.dumps(res, indent=1))
        print("listo", experimentos.etiqueta(cfg), {k: round(v["exacta"], 3) for k, v in res["eval"].items()})
    print("resultados en", carpeta)
