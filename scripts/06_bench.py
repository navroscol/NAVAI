"""Benchmark de un preset en el dispositivo indicado (tokens sintéticos). Imprime una línea
'RESULTADO {json}'. Se ejecuta en un proceso aparte por configuración: si una se queda sin
memoria, las demás siguen.

    python scripts/06_bench.py --json '{"preset": "navros-1b", "device": "xla", "precision": "bf16", ...}'
"""
import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    kw = json.loads(ap.parse_args().json)
    try:
        from navros.lm import benchmark
        res = benchmark(log=lambda s: print(s, flush=True), **kw)
    except Exception:
        res = dict(kw, error=traceback.format_exc()[-1500:])
    print("RESULTADO " + json.dumps(res, default=str), flush=True)


if __name__ == "__main__":
    main()
