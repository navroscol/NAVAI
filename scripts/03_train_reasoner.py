"""Entrena un razonador (bucle o fijo) y evalúa generalización de longitud.

    python scripts/03_train_reasoner.py --task suma --arch bucle --L 8 --steps 3000
    python scripts/03_train_reasoner.py --task suma --arch fijo  --L 8 --steps 3000
"""
import argparse
import sys
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navros.reasoner import RunCfg, save, train  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    for f in fields(RunCfg):
        if f.name == "eval_mult":
            ap.add_argument("--eval_mult", type=int, nargs="+", default=list(f.default))
        elif f.type in ("bool", bool):
            ap.add_argument(f"--{f.name}", type=lambda s: s.lower() in ("1", "true", "yes"), default=f.default)
        else:
            ap.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
    ap.add_argument("--out", default="")
    a = vars(ap.parse_args())
    out = a.pop("out")
    a["eval_mult"] = tuple(a["eval_mult"])
    rc = RunCfg(**a)
    res = train(rc)
    save(res, out or f"results/reasoner/{rc.task}_{rc.arch}_L{rc.L}_d{rc.d}_s{rc.seed}{('_' + rc.tag) if rc.tag else ''}.json")


if __name__ == "__main__":
    main()
