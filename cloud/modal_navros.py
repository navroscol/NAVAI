"""NAVROS en Modal (H100): benchmark, corpus y entrenamiento del modelo de 1B.

    modal run cloud/modal_navros.py::bench                 # ~5 min de H100: velocidad y memoria reales
    modal run cloud/modal_navros.py::datos                 # CPU: corpus bilingüe en el Volume navros-data
    modal run --detach cloud/modal_navros.py::entrenar     # tramo de entrenamiento; reanuda solo

Volúmenes:
  navros-data  → /data  (tokenizer.json, manifest.json, shards uint16)
  navros-ckpt  → /ckpt  (checkpoints y resultados JSON)
El código se monta desde este repositorio en cada ejecución (lo que corre es lo que hay aquí).
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/root/navros-ai"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.10.0", "numpy", "tokenizers", "datasets")
    .add_local_dir(ROOT / "navros", f"{REMOTE}/navros", ignore=["**/__pycache__/**"])
    .add_local_dir(ROOT / "scripts", f"{REMOTE}/scripts", ignore=["**/__pycache__/**"])
)
app = modal.App("navros", image=image)
data_vol = modal.Volume.from_name("navros-data", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("navros-ckpt", create_if_missing=True)


def _setup():
    sys.path.insert(0, REMOTE)
    import torch
    info = dict(torch=torch.__version__, cuda=torch.cuda.is_available(),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, cpus=os.cpu_count())
    print(info, flush=True)
    return info


@app.function(gpu="H100", cpu=8, memory=65536, timeout=45 * 60)
def bench(configs_json: str = ""):
    """Velocidad y memoria del entrenador real (lm_ddp, un proceso) con tokens sintéticos.
    configs_json: lista JSON de configuraciones (por defecto: 1B recurrente y 1B fijo, micro 8)."""
    configs = json.loads(configs_json) if configs_json else None
    info = _setup()
    from navros.lm import LMRun
    from navros.lm_ddp import train_ddp
    base = dict(synthetic=True, T=1024, batch=64, warmup=2, precision="bf16", device="cuda", grad_ckpt=False,
                eval_every=10**9, eval_seq=8, r_eval=[1, 2, 4], log_every=1, muon_buf="fp32")
    configs = configs or [dict(preset="navros-1b", micro=8), dict(preset="navros-1b-fix", micro=8)]
    out = []
    for c in configs:
        kw = base | c | dict(tokens=8 * base["batch"] * base["T"], tag="bench-" + c["preset"])
        try:
            res = train_ddp(LMRun(**kw), ns_dtype="bf16", log=lambda s: print(s, flush=True))
            rows = [h for h in res["history"] if "tok_s" in h][2:]  # sin los pasos de arranque
            out.append(dict(c, tok_s=sum(h["tok_s"] for h in rows) / len(rows), tflops=sum(h["tflops"] for h in rows) / len(rows),
                            mem_gb=max(h["mem_gb"] for h in rows), flops_per_token=res["flops_per_token"]))
        except Exception as e:  # p. ej. sin memoria: se registra y se sigue
            out.append(dict(c, error=repr(e)[:500]))
        print("BENCH", json.dumps(out[-1]), flush=True)
        import torch
        torch.cuda.empty_cache()
    return dict(env=info, results=out)


@app.function(cpu=8, memory=32768, timeout=24 * 3600, volumes={"/data": data_vol})
def datos(tokens_per_lang: int = 3_000_000_000, part: int = 0, n_parts: int = 4):
    """Corpus bilingüe con el tokenizador de Kaggle (subido a /data/tokenizer_kaggle.json)."""
    _setup()
    from navros.dataprep import prepare
    tok = "/data/tokenizer_kaggle.json"
    out = "/data/navros_data" if part == 0 else f"/data/navros_data_p{part}"
    m = prepare(out, target_tokens_per_lang=tokens_per_lang, time_budget_s=22 * 3600, part=(part, n_parts),
                tokenizer_path=tok if os.path.exists(tok) else None, log=lambda s: print(s, flush=True))
    data_vol.commit()
    return {l: v["train"]["tokens"] for l, v in m["langs"].items()}


@app.function(gpu="H100", cpu=8, memory=65536, timeout=24 * 3600, volumes={"/data": data_vol, "/ckpt": ckpt_vol},
              retries=modal.Retries(max_retries=3, initial_delay=10.0))
def entrenar(horas: float = 3.0, preset: str = "navros-1b", total_tokens: int = 10_000_000_000,
             micro: int = 8, tag: str = "navros-1b-v1"):
    """Un tramo de entrenamiento de `horas`. Reanuda del último checkpoint de /ckpt/<tag>.
    Si Modal interrumpe la máquina, el reintento reanuda desde el último checkpoint confirmado."""
    _setup()
    from navros.lm import LMRun
    from navros.lm_ddp import train_ddp
    # copia local de los datos (lecturas aleatorias rápidas; el Volume es de red)
    t = time.time()
    local = "/tmp/navros_data"
    if not os.path.exists(local):
        shutil.copytree("/data/navros_data", local)
    print(f"datos copiados en {time.time() - t:.0f}s", flush=True)
    rc = LMRun(preset=preset, data_dirs=[local], T=1024, batch=256, micro=micro, tokens=total_tokens,
               lr_muon=0.02, lr_adam=1e-3, warmup=200, decay_frac=0.2, precision="bf16", device="cuda",
               grad_ckpt=False, muon_buf="fp32", eval_every=200, eval_seq=64, r_eval=(1, 2, 3, 4, 6, 8, 12, 16),
               log_every=10, ckpt_dir=f"/ckpt/{tag}", ckpt_every_min=20, time_budget_s=horas * 3600, tag=tag)
    res = train_ddp(rc, ns_dtype="bf16", log=lambda s: print(s, flush=True), on_save=ckpt_vol.commit)
    Path(f"/ckpt/{tag}/resultados").mkdir(parents=True, exist_ok=True)
    Path(f"/ckpt/{tag}/resultados/tramo_{int(time.time())}.json").write_text(json.dumps(res, default=float))
    ckpt_vol.commit()
    return dict(step=res["step"], finished=res["finished"], last=res["history"][-3:])


@app.local_entrypoint()
def main():
    print(bench.remote())
