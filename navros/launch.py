"""Lanza N procesos de entrenamiento (uno por GPU) sin torchrun: dirección de encuentro
explícita en 127.0.0.1, que evita problemas de resolución de nombres. Devuelve el código
de salida del rango 0 y reenvía su salida en vivo."""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from pathlib import Path


def launch(cfg: dict, out: str, nproc: int, root: str, ns_dtype: str = "fp16", log_dir: str | None = None) -> int:
    port = str(29500 + random.randint(0, 999))
    procs = []
    log_dir = Path(log_dir or Path(out).parent)
    log_dir.mkdir(parents=True, exist_ok=True)
    for r in range(nproc):
        env = dict(os.environ, MASTER_ADDR="127.0.0.1", MASTER_PORT=port, WORLD_SIZE=str(nproc), RANK=str(r),
                   LOCAL_RANK=str(r), PYTHONUNBUFFERED="1", PYTORCH_ALLOC_CONF="expandable_segments:True")
        cmd = [sys.executable, str(Path(root) / "scripts" / "05_train_lm.py"), "--json", json.dumps(cfg),
               "--ns_dtype", ns_dtype, "--out", out]
        stdout = None if r == 0 else open(log_dir / f"{Path(out).stem}.r{r}.log", "w")
        procs.append(subprocess.Popen(cmd, env=env, stdout=stdout, stderr=subprocess.STDOUT if r else None))
    codes = [p.wait() for p in procs]
    if any(codes):
        print("códigos de salida por rango:", codes, flush=True)
    return codes[0]
