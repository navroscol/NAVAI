"""Más tokens para NAVROS-1B: preentrenamiento continuado sobre texto nuevo (workspace heraclutes).

    modal run --detach cloud/modal_pre.py::datos          # CPU: corpus fresco (partición 1/4)
    modal run --detach cloud/modal_pre.py::entrenar       # H100: sigue entrenando el 1B

No reanuda el entrenamiento anterior: aquel terminó con el LR ya decaído a cero, así que esto es
una corrida nueva que parte de sus pesos, con su propio calentamiento y decaimiento. El texto es
de una partición distinta del stream, de modo que el modelo no vuelve a leer lo mismo.
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
    .pip_install("torch==2.10.0", "numpy", "tokenizers", "datasets")
    .add_local_dir(ROOT / "navros", f"{REMOTE}/navros", ignore=["**/__pycache__/**"])
)
app = modal.App("navros-pre", image=image)
vol = modal.Volume.from_name("navros-data", create_if_missing=True)


def _setup():
    sys.path.insert(0, REMOTE)
    import torch
    print(dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               cpus=os.cpu_count()), flush=True)


def _sha(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while c := f.read(block):
            h.update(c)
    return h.hexdigest()


@app.function(cpu=8, memory=16384, timeout=12 * 3600, volumes={"/data": vol})
def datos(tokens_per_lang: int = 1_500_000_000, part: int = 1, n_parts: int = 4, salida: str = "/data/corpus_p1",
          eval_de: str = ""):
    """Corpus bilingüe nuevo con el tokenizador del repo. part=1 evita el texto que ya vio.

    `prepare` solo crea val y test en la partición 0, así que para las demás hay que traerlos de
    otro corpus con `eval_de` (ruta a su manifest.json). Usar los mismos conjuntos que el modelo
    base tiene además la ventaja de que las pérdidas son comparables entre corridas."""
    _setup()
    import shutil
    from navros.dataprep import prepare
    m = prepare(salida, target_tokens_per_lang=tokens_per_lang, time_budget_s=10 * 3600,
                part=(part, n_parts), tokenizer_path=f"{REMOTE}/navros/assets/tokenizer_navros_32k.json",
                log=lambda s: print(s, flush=True))
    if part != 0:
        assert eval_de, "una partición > 0 necesita val/test de otro corpus: pasa eval_de"
        origen = json.loads(Path(eval_de).read_text())
        base = Path(eval_de).parent
        for lang, v in origen["langs"].items():
            for split in ("val", "test"):
                dst = Path(salida) / v[split]["file"]
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(base / v[split]["file"], dst)
                m["langs"][lang][split] = v[split]
        m["eval_de"] = eval_de
        (Path(salida) / "manifest.json").write_text(json.dumps(m, indent=1, ensure_ascii=False))
        print(f"val/test copiados de {eval_de}", flush=True)
    vol.commit()
    return {l: v["train"]["tokens"] for l, v in m["langs"].items()}


@app.function(gpu="H100", cpu=4, memory=32768, timeout=12 * 3600, volumes={"/data": vol},
              retries=modal.Retries(max_retries=2, initial_delay=10.0))
def entrenar(horas: float = 4.0, total_tokens: int = 800_000_000, lr_muon: float = 0.01,
             micro: int = 8, tag: str = "navros-1b-mas-tokens", datos_dir: str = "/data/corpus_p1"):
    """Sigue entrenando el 1B con texto nuevo. Reanuda solo si ya hay checkpoint de este tag."""
    _setup()
    import shutil
    import torch
    from navros.lm import LMRun
    from navros.lm_ddp import train_ddp

    base = "/tmp/base_bf16.pt"
    if not os.path.exists(base):
        subprocess.run(["curl", "-sSL", "--retry", "10", "--retry-all-errors", "-o", base, BASE_URL], check=True)
        assert _sha(base) == BASE_SHA, "los pesos base no coinciden con el SHA-256 publicado"
        print("pesos base verificados", flush=True)

    local = "/tmp/corpus"                      # copia local: el Volume es de red y aquí hay lectura aleatoria
    if not os.path.exists(local):
        t0 = time.time()
        shutil.copytree(datos_dir, local)
        print(f"corpus copiado en {time.time()-t0:.0f}s", flush=True)

    rc = LMRun(preset="navros-1b-fix", data_dirs=[local], init_from=base, T=1024, batch=256, micro=micro,
               tokens=total_tokens, lr_muon=lr_muon, lr_adam=lr_muon / 20, warmup=100, decay_frac=0.25,
               precision="bf16", device="cuda", muon_buf="fp32", grad_ckpt=False,
               eval_every=200, eval_seq=64, log_every=10, ckpt_dir=f"/data/{tag}", ckpt_every_min=25,
               time_budget_s=horas * 3600, tag=tag)
    res = train_ddp(rc, ns_dtype="bf16", log=lambda s: print(s, flush=True), on_save=vol.commit)

    sd = {n: p.detach().to(torch.bfloat16).cpu() for n, p in
          torch.load(f"/data/{tag}/opt_r0.pt", map_location="cpu", weights_only=False)["masters"].items()}
    out = Path(f"/data/{tag}/export")
    out.mkdir(parents=True, exist_ok=True)
    torch.save(dict(step=res["step"], preset="navros-1b-fix", tag=tag, vocab=32768, state=sd), out / "pesos_bf16.tmp")
    os.replace(out / "pesos_bf16.tmp", out / "pesos_bf16.pt")
    info = dict(step=res["step"], finished=res["finished"], val=res.get("val"), test=res.get("test"),
                tokens_vistos=res["step"] * rc.batch * rc.T, sha256=_sha(out / "pesos_bf16.pt"))
    (out / "pesos_bf16.json").write_text(json.dumps(info))
    (Path(f"/data/{tag}") / "resultados.json").write_text(json.dumps(res, default=float))
    vol.commit()
    print("EXPORT", json.dumps(info), flush=True)
    return info
