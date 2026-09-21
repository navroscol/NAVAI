"""Porte de Phi-4-mini al formato de NAVROS (tokenizador propio de 32K), en Modal, con verificación.

    modal run cloud/modal_porte_phi.py --modelo microsoft/Phi-4-mini-instruct \
        --donante-url https://github.com/navroscol/navros-ai/releases/download/pesos-1b-2600M/<archivo>.pt
    modal run cloud/modal_porte_phi.py --modelo microsoft/Phi-4-mini-reasoning --donante-gcs base/navros1b_2600M.pt

Hace, en un contenedor de CPU (unos 30 min, ~1 $):
  1. descarga el modelo de Hugging Face (MIT) y el DONANTE (NAVROS-1B, que usa nuestro tokenizador);
  2. mapea el cuerpo a los nombres de NAVROS (navros/porte_phi.py) y lo verifica contra
     transformers con la embedding original: los logits deben coincidir;
  3. trasplanta la embedding de 200K a nuestro tokenizador de 32K por OMP (arXiv 2506.06607):
     tokens compartidos copiados, el resto como combinación k-dispersa aprendida en el donante;
  4. deja el export en GCS (navros-cadena/base/<nombre>.pt) y en el Volume navros-cadena-ckpt,
     listo para `modal run cloud/modal_cadena.py::tramo --preset phi4mini --base-gcs base/<nombre>.pt`.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/root/navros-ai"
BUCKET = os.environ.get("NAVROS_BUCKET", "navros-respaldo-saasvareoz")
CADENA = "/gcs/navros-cadena"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.10.0", "numpy", "tokenizers", "safetensors", "transformers>=4.51", "huggingface_hub")
    .add_local_dir(ROOT / "navros", f"{REMOTE}/navros", ignore=["**/__pycache__/**"])
)
app = modal.App("navros-porte-phi", image=image)
vol = modal.Volume.from_name("navros-cadena-ckpt", create_if_missing=True)
gcs = modal.CloudBucketMount(BUCKET, secret=modal.Secret.from_name("gcs-navros"))


def _sha(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while c := f.read(block):
            h.update(c)
    return h.hexdigest()


@app.function(cpu=8, memory=65536, timeout=3 * 3600, volumes={"/ckpt": vol, "/gcs": gcs})
def portar(modelo: str = "microsoft/Phi-4-mini-instruct", verificar: int = 2, donante_url: str = "",
           donante_sha: str = "", donante_gcs: str = "", metodo: str = "omp", k: int = 64) -> dict:
    sys.path.insert(0, REMOTE)
    from huggingface_hub import snapshot_download
    t0 = time.time()
    hf_dir = snapshot_download(modelo, allow_patterns=["*.json", "*.safetensors"])
    print(f"descargado {modelo} en {time.time() - t0:.0f}s: {hf_dir}", flush=True)
    donante = ""
    if metodo == "omp":
        donante = "/tmp/donante.pt"
        if donante_gcs:
            import shutil
            shutil.copyfile(Path(CADENA) / donante_gcs, donante)
        else:
            assert donante_url, "--metodo omp necesita --donante-url (release de NAVROS-1B) o --donante-gcs"
            subprocess.run(["curl", "-sSL", "--retry", "10", "--retry-all-errors", "-o", donante, donante_url], check=True)
        if donante_sha:
            assert _sha(donante) == donante_sha, "el donante no coincide con el SHA-256 dado"
        print(f"donante listo ({Path(donante).stat().st_size / 2**30:.2f} GiB)", flush=True)
    nombre = modelo.split("/")[-1].lower().replace("-", "_") + "_navros32k.pt"
    salida = Path("/ckpt/base") / nombre
    salida.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "navros.porte_phi", "--hf", hf_dir, "--salida", str(salida), "--verificar", str(verificar),
           "--metodo", metodo, "--k", str(k)] + (["--donante", donante] if donante else [])
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REMOTE, check=True)
    vol.commit()
    sha = _sha(salida)
    dst = Path(CADENA) / "base" / nombre
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp")
    import shutil
    shutil.copyfile(salida, tmp)
    os.replace(tmp, dst)
    info = dict(modelo=modelo, export_volume=str(salida), export_gcs=f"base/{nombre}", sha256=sha, metodo=metodo, k=k,
                donante=donante_gcs or donante_url, GiB=round(salida.stat().st_size / 2**30, 2), segundos=round(time.time() - t0))
    (dst.with_suffix(".json")).write_text(json.dumps(info, indent=1))
    print("PORTE", json.dumps(info), flush=True)
    return info


@app.local_entrypoint()
def main(modelo: str = "microsoft/Phi-4-mini-instruct", verificar: int = 2, donante_url: str = "", donante_sha: str = "",
         donante_gcs: str = "", metodo: str = "omp", k: int = 64):
    print(portar.remote(modelo, verificar, donante_url, donante_sha, donante_gcs, metodo, k))
