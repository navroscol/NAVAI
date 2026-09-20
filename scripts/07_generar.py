"""Genera continuaciones de un conjunto fijo de frases con un export de pesos de NAVROS.

    python scripts/07_generar.py --pesos pesos_bf16.pt --out results/generacion
Comprueba antes que la caché KV reproduce los logits del forward completo.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tokenizers import Tokenizer  # noqa: E402

from navros.generate import EOT, load_export, run_prompts, to_markdown, verify_cache  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pesos", required=True)
ap.add_argument("--tokenizer", default=str(Path(__file__).resolve().parents[1] / "navros/assets/tokenizer_navros_32k.json"))
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
ap.add_argument("--n_new", type=int, default=120)
ap.add_argument("--out", default="results/generacion")
a = ap.parse_args()

t0 = time.time()
dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[a.dtype]
model, meta = load_export(a.pesos, a.device, dtype)
tok = Tokenizer.from_file(a.tokenizer)
print(f"cargado: paso {meta['step']} · {meta['preset']} · {sum(p.numel() for p in model.parameters()):,} params · "
      f"{a.device} {a.dtype} ({time.time() - t0:.0f}s)", flush=True)

ids = [EOT] + tok.encode("La inteligencia artificial es una rama de la informática que estudia cómo").ids
d, scale = verify_cache(model, ids + list(range(100, 140)))
print(f"verificación caché KV: max |Δlogits| = {d:.2e} (|logits| máx {scale:.1f})", flush=True)
assert d <= 1e-3 * max(1.0, scale) if a.dtype == "fp32" else d <= 5e-2 * max(1.0, scale), "la caché KV no coincide"

t1 = time.time()
rows = run_prompts(model, tok, n_new=a.n_new, log=lambda s: print(s, flush=True))
meta.update(cache_check=dict(max_abs_diff=d, max_abs_logit=scale), device=a.device, dtype=a.dtype,
            seconds=time.time() - t1)
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
name = f"paso_{meta['step']:05d}"
(out / f"{name}.json").write_text(json.dumps(dict(meta=meta, rows=rows), indent=1, ensure_ascii=False))
(out / f"{name}.md").write_text(to_markdown(rows, meta))
print(f"\nguardado en {out}/{name}.md ({time.time() - t1:.0f}s de generación)", flush=True)
