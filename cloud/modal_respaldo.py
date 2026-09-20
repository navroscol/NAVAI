"""Respaldo de NAVROS en Google Cloud Storage, montado dentro de Modal (sin pasar por el portátil).

    modal run cloud/modal_respaldo.py::respaldar                  # lo importante y pequeño
    modal run cloud/modal_respaldo.py::respaldar --con-optimizador   # + estado para reanudar

Qué se copia por defecto: pesos exportados, manifiestos, resultados y charlas. El estado del
optimizador (8 GiB por checkpoint) solo con `--con-optimizador`, porque únicamente sirve para
reanudar ese entrenamiento concreto.

Requisitos (los configura la persona dueña de la cuenta, no este código):
  1. Un bucket en GCS, p. ej. `navros-respaldo` (Standard si se lee a veces, Nearline si es archivo).
  2. Una cuenta de servicio con «Storage Object Admin» sobre ese bucket, y su clave JSON.
  3. En Modal → Secrets, un secreto llamado `gcs-navros` con la clave `SERVICE_ACCOUNT_JSON`
     y el contenido del JSON pegado como valor.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import modal

BUCKET = os.environ.get("NAVROS_BUCKET", "navros-respaldo-saasvareoz")
IMPORTANTES = ("pesos_bf16.pt", "pesos_bf16.json", "manifest.json", "resultados.json",
               "charlas.json", "latest.json", "human_chat.txt", ".bin", ".msk")
OPTIMIZADOR = ("opt_r0.pt", "model.pt")

app = modal.App("navros-respaldo")
image = modal.Image.debian_slim(python_version="3.12")
sft_vol = modal.Volume.from_name("navros-sft")
gcs = modal.CloudBucketMount(BUCKET, secret=modal.Secret.from_name("gcs-navros"))


def _sha(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while c := f.read(block):
            h.update(c)
    return h.hexdigest()


@app.function(image=image, cpu=2, memory=8192, timeout=6 * 3600,
              volumes={"/sft": sft_vol, "/gcs": gcs})
def respaldar(con_optimizador: bool = False, origen: str = "/sft", destino: str = "/gcs/navros-sft"):
    """Copia lo que haya cambiado, comparando tamaño y SHA-256 contra el índice del bucket."""
    org, dst = Path(origen), Path(destino)
    dst.mkdir(parents=True, exist_ok=True)
    indice_path = dst / "indice.json"
    indice = json.loads(indice_path.read_text()) if indice_path.exists() else {}
    quiero = IMPORTANTES + (OPTIMIZADOR if con_optimizador else ())

    copiados, saltados, bytes_ = [], 0, 0
    for p in sorted(org.rglob("*")):
        if not p.is_file() or not any(str(p).endswith(q) for q in quiero):
            continue
        rel = str(p.relative_to(org))
        tam = p.stat().st_size
        previo = indice.get(rel)
        if previo and previo["bytes"] == tam and previo["mtime"] == int(p.stat().st_mtime):
            saltados += 1
            continue
        t0 = time.time()
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dst / rel)
        indice[rel] = dict(bytes=tam, mtime=int(p.stat().st_mtime), sha256=_sha(p), copiado=int(time.time()))
        copiados.append(rel)
        bytes_ += tam
        print(f"  ↑ {rel} ({tam/1e6:.1f} MB en {time.time()-t0:.0f}s)", flush=True)

    indice_path.write_text(json.dumps(indice, indent=1))
    resumen = dict(bucket=BUCKET, destino=destino, copiados=len(copiados), saltados=saltados,
                   MB=round(bytes_ / 1e6, 1), archivos=copiados[:20])
    print("RESPALDO", json.dumps(resumen), flush=True)
    return resumen


@app.function(image=image, cpu=1, memory=2048, timeout=900, volumes={"/gcs": gcs})
def listar(destino: str = "/gcs/navros-sft"):
    """Qué hay respaldado y cuánto ocupa (para vigilar el coste de GCS)."""
    d = Path(destino)
    if not d.exists():
        return dict(existe=False)
    archivos = [(str(p.relative_to(d)), p.stat().st_size) for p in d.rglob("*") if p.is_file()]
    total = sum(t for _, t in archivos)
    print(f"{len(archivos)} archivos · {total/2**30:.2f} GiB · ≈ {total/2**30*0.02:.2f} $/mes en Standard", flush=True)
    for n, t in sorted(archivos, key=lambda x: -x[1])[:15]:
        print(f"  {t/1e6:9.1f} MB  {n}", flush=True)
    return dict(archivos=len(archivos), GiB=round(total / 2 ** 30, 2))
