"""Ajuste por conversaciones de NAVROS-1B en Modal (workspace heraclutes).

    modal run cloud/modal_sft.py::datos                      # CPU: construye el conjunto de chat
    modal run --detach cloud/modal_sft.py::entrenar          # H100: ajusta desde los pesos base

Los pesos de partida se bajan de la release pública de GitHub, así que no hace falta mover nada
entre workspaces. Volumen `navros-sft` → /sft (datos, checkpoints y export final).
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
BASE_URL = ("https://github.com/navroscol/navros-ai/releases/download/"
            "pesos-1b-fix-paso-03623/pesos_bf16.pt")
BASE_SHA = "a370c6bec0e9e8e1f2bc960bc70f926a19b86e917c0f69301de256299cac246a"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl")
    .pip_install("torch==2.10.0", "numpy", "tokenizers", "datasets", "zstandard")
    .add_local_dir(ROOT / "navros", f"{REMOTE}/navros", ignore=["**/__pycache__/**"])
    .add_local_dir(ROOT / "scripts", f"{REMOTE}/scripts", ignore=["**/__pycache__/**"])
)
app = modal.App("navros-sft", image=image)
vol = modal.Volume.from_name("navros-sft", create_if_missing=True)


def _setup():
    sys.path.insert(0, REMOTE)
    import torch
    info = dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                cpus=os.cpu_count())
    print(info, flush=True)
    return info


def _sha(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while c := f.read(block):
            h.update(c)
    return h.hexdigest()


def _tokenizador():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(f"{REMOTE}/navros/assets/tokenizer_navros_32k.json")


@app.function(cpu=4, memory=16384, timeout=6 * 3600, volumes={"/sft": vol})
def datos(salida: str = "/sft/datos"):
    """Descarga las fuentes de conversación, las tokeniza con máscara y las empaqueta."""
    _setup()
    from navros.sft import construir, plan
    t0 = time.time()
    kaggle = "/sft/kaggle" if os.path.exists("/sft/kaggle/human_chat.txt") else None
    m = construir(salida, _tokenizador(), plan(kaggle_dir=kaggle), log=lambda s: print(s, flush=True))
    vol.commit()
    print(f"listo en {(time.time()-t0)/60:.1f} min", flush=True)
    return {l: dict(tokens=v["train"]["tokens"], convs=v["conversaciones"]) for l, v in m["langs"].items()}


@app.function(gpu="H100", cpu=4, memory=32768, timeout=6 * 3600, volumes={"/sft": vol},
              retries=modal.Retries(max_retries=2, initial_delay=10.0))
def entrenar(epocas: float = 2.0, horas: float = 1.0, lr_muon: float = 0.004, micro: int = 8,
             batch: int = 64, tag: str = "navros-1b-chat-v1", datos_dir: str = "/sft/datos"):
    """Ajuste fino sobre conversaciones, partiendo de los pesos publicados del modelo base."""
    _setup()
    import torch
    from navros.lm import LMRun
    from navros.lm_ddp import train_ddp

    base = "/tmp/base_bf16.pt"
    if not os.path.exists(base):
        t0 = time.time()
        subprocess.run(["curl", "-sSL", "--retry", "10", "--retry-all-errors", "-o", base, BASE_URL], check=True)
        assert _sha(base) == BASE_SHA, "los pesos base no coinciden con el SHA-256 publicado"
        print(f"pesos base descargados y verificados en {time.time()-t0:.0f}s", flush=True)

    m = json.loads((Path(datos_dir) / "manifest.json").read_text())
    tokens_dataset = sum(v["train"]["tokens"] for v in m["langs"].values())
    total = int(epocas * tokens_dataset)
    peso_es = m["langs"]["es"]["train"]["tokens"] / tokens_dataset      # mezcla según lo que hay
    print(f"conjunto: {tokens_dataset/1e6:.1f}M tokens ({100*peso_es:.0f}% es) · "
          f"{epocas} épocas = {total/1e6:.1f}M tokens", flush=True)

    rc = LMRun(preset="navros-1b-fix", sft_dir=datos_dir, init_from=base,
               langs={"es": round(peso_es, 3), "en": round(1 - peso_es, 3)},
               T=1024, batch=batch, micro=micro, tokens=total,
               lr_muon=lr_muon, lr_adam=lr_muon / 20, warmup=20, decay_frac=0.5,
               precision="bf16", device="cuda", muon_buf="fp32", grad_ckpt=False,
               eval_every=50, eval_seq=32, log_every=10,
               ckpt_dir=f"/sft/{tag}", ckpt_every_min=20, time_budget_s=horas * 3600, tag=tag)
    res = train_ddp(rc, ns_dtype="bf16", log=lambda s: print(s, flush=True), on_save=vol.commit)

    # export bf16 para publicarlo y usarlo en local
    sd = {n: p.detach().to(torch.bfloat16).cpu() for n, p in
          torch.load(f"/sft/{tag}/opt_r0.pt", map_location="cpu", weights_only=False)["masters"].items()}
    out = Path(f"/sft/{tag}/export")
    out.mkdir(parents=True, exist_ok=True)
    torch.save(dict(step=res["step"], preset="navros-1b-fix", tag=tag, vocab=32768, state=sd), out / "pesos_bf16.tmp")
    os.replace(out / "pesos_bf16.tmp", out / "pesos_bf16.pt")
    info = dict(step=res["step"], finished=res["finished"], val=res.get("val"), test=res.get("test"),
                sha256=_sha(out / "pesos_bf16.pt"), bytes=(out / "pesos_bf16.pt").stat().st_size)
    (out / "pesos_bf16.json").write_text(json.dumps(info))
    (Path(f"/sft/{tag}") / "resultados.json").write_text(json.dumps(res, default=float))
    vol.commit()
    print("EXPORT", json.dumps(info), flush=True)
    return info


@app.function(gpu="H100", cpu=4, memory=32768, timeout=3600, volumes={"/sft": vol})
def charlar(tag: str = "navros-1b-chat-v1", n_new: int = 120):
    """Conversaciones de prueba con el modelo ajustado, en el formato de chat."""
    _setup()
    import torch
    from navros.generate import Sesion, load_export
    modelo, meta = load_export(f"/sft/{tag}/export/pesos_bf16.pt", "cuda", torch.float32)
    tok = _tokenizador()
    guiones = [["Hola, ¿qué tal?", "¿Y tú a qué te dedicas?"],
               ["Estoy aburrido, ¿qué puedo hacer esta tarde?", "Me gusta más leer, ¿algún libro?"],
               ["Explícame como si tuviera diez años qué es la gravedad", "¿Y por qué no nos caemos hacia el sol?"],
               ["Hi! How are you doing today?", "What do you like to do for fun?"],
               ["I had a rough day at work.", "Thanks, that helps."]]
    salida = []
    for guion in guiones:
        ses, dialogo = Sesion(modelo), []
        for turno in guion:
            ids = ([0] if ses.pos == 0 else []) + tok.encode(f"Usuario: {turno}\nAsistente: ").ids
            if len(ids) > ses.libre:
                break
            ses.feed(ids)
            trozos = list(ses.stream(n_new=n_new, temperature=0.7, top_p=0.9, seed=len(dialogo), repetition_penalty=1.1))
            respuesta = tok.decode(trozos).split("Usuario:")[0].strip()
            dialogo.append(dict(usuario=turno, asistente=respuesta))
            print(f"\nUsuario: {turno}\nAsistente: {respuesta}", flush=True)
        salida.append(dialogo)
    Path(f"/sft/{tag}/charlas.json").write_text(json.dumps(dict(meta=meta, dialogos=salida), ensure_ascii=False, indent=1))
    vol.commit()
    return salida
