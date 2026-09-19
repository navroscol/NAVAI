"""Exportar pesos de NAVROS desde el Volume de checkpoints (para publicarlos como release de GitHub).

    modal run cloud/modal_pesos.py::exportar            # CPU, ~2 min: /ckpt/<tag>/export/pesos_bf16.pt
    modal volume get navros-ckpt <tag>/export/ .         # descarga local (2,1 GB para el 1B)
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import modal

app = modal.App("navros-pesos")
ckpt_vol = modal.Volume.from_name("navros-ckpt")
torch_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch==2.10.0", "numpy", index_url="https://download.pytorch.org/whl/cpu")


def _sha256(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


@app.function(image=torch_image, cpu=2, memory=8192, timeout=30 * 60, volumes={"/ckpt": ckpt_vol})
def exportar(tag: str = "navros-1b-fix-v1"):
    """Maestros fp32 del último checkpoint confirmado → pesos bf16 (sin estado del optimizador)."""
    import torch
    d = Path(f"/ckpt/{tag}")
    meta = json.loads((d / "latest.json").read_text())
    assert meta["world"] == 1, "este export asume un único rango (todos los maestros en opt_r0.pt)"
    so = torch.load(d / "opt_r0.pt", map_location="cpu", mmap=True, weights_only=False)
    state = {n: t.to(torch.bfloat16) for n, t in so["masters"].items()}
    n_params = sum(t.numel() for t in state.values())
    out = d / "export"
    out.mkdir(exist_ok=True)
    tmp = out / "pesos_bf16.tmp"
    torch.save(dict(step=meta["step"], preset=meta["preset"], tag=tag, vocab=state["emb"].shape[0], state=state), tmp)
    os.replace(tmp, out / "pesos_bf16.pt")
    info = dict(step=meta["step"], preset=meta["preset"], tag=tag, params=n_params,
                bytes=(out / "pesos_bf16.pt").stat().st_size, sha256=_sha256(out / "pesos_bf16.pt"))
    (out / "pesos_bf16.json").write_text(json.dumps(info))
    ckpt_vol.commit()
    print(info, flush=True)
    return info

