"""Entrena NAVROS-LM en 1 o varias GPU. Con varias, lanzar con torchrun:

    python -m torch.distributed.run --standalone --nproc_per_node=2 scripts/05_train_lm.py \
        --json '{"preset": "navros-1b", "T": 1024, "batch": 256, "micro": 2, ...}' --out results/lm.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navros.lm import LMRun  # noqa: E402
from navros.lm_ddp import train_ddp  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--ns_dtype", default="fp16")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    rc = LMRun(**json.loads(a.json))
    res = train_ddp(rc, ns_dtype=None if a.ns_dtype == "none" else a.ns_dtype, log=lambda s: print(s, flush=True))
    if int(os.environ.get("RANK", 0)) == 0 and a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
