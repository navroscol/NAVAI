"""Paso 1 del protocolo: gradientes del oráculo contra diferencias finitas (float64).

    python scripts/01_gradcheck.py            # tabla completa
    python scripts/01_gradcheck.py --quiet    # solo el resumen

Criterio: para cada tensor, el mejor error relativo del barrido de epsilon de su
derivada direccional debe quedar por debajo de 1e-4.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navros.config import NavrosConfig  # noqa: E402
from navros.oracle.gradcheck import directional_check, entrywise_check  # noqa: E402
from navros.oracle.model import NavrosNP  # noqa: E402

TOL = 1e-4

CASES = [
    # nombre, config, B, T, r, k, padding
    ("lm_fijo_causal", NavrosConfig(vocab=23, d=16, n_heads=2, ffn=24, n_pre=2, n_core=0, causal=True), 3, 7, None, None, False),
    ("razonador_bucle_bptt_completo", NavrosConfig(vocab=13, d=16, n_heads=2, ffn=24, n_core=1, abacus=11), 3, 6, 5, None, True),
    ("razonador_bucle_bptt_truncado", NavrosConfig(vocab=13, d=16, n_heads=2, ffn=24, n_core=2, abacus=11), 3, 6, 6, 2, True),
    ("referencia_fija_3_bloques", NavrosConfig(vocab=13, d=16, n_heads=4, ffn=24, n_core=2, n_blocks=3, abacus=11), 2, 6, 3, None, True),
    ("lm_recurrente_preludio_coda", NavrosConfig(vocab=19, d=16, n_heads=2, ffn=24, n_pre=1, n_core=2, n_coda=1, causal=True), 2, 8, 4, 2, False),
    ("sin_rope_con_ábaco", NavrosConfig(vocab=13, d=12, n_heads=3, ffn=20, n_core=1, rope=False, abacus=9), 3, 5, 3, None, False),
    ("rope_posiciones_aleatorias", NavrosConfig(vocab=13, d=16, n_heads=2, ffn=24, n_core=1), 3, 6, 4, 2, "pos"),
]


def make_batch(cfg, B, T, rng, pad):
    batch = dict(
        tokens=rng.integers(0, cfg.vocab, (B, T)),
        targets=rng.integers(0, cfg.vocab, (B, T)),
        weights=rng.random((B, T)) * (rng.random((B, T)) > 0.25),
    )
    batch["weights"][0, 0] = 1.0
    if cfg.abacus:
        batch["abacus"] = rng.integers(0, cfg.abacus, (B, T))
    if pad == "pos":
        batch["pos"] = np.stack([np.sort(rng.choice(40, T, replace=False)) for _ in range(B)])
    elif pad:
        valid = np.ones((B, T), dtype=bool)
        valid[0, -2:] = False
        valid[-1, -1:] = False
        batch["valid"] = valid
        batch["weights"] = batch["weights"] * valid
    return batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--out", default="results/gradcheck.json")
    args = ap.parse_args()

    worst_all, summary = 0.0, []
    for i, (name, cfg, B, T, r, k, pad) in enumerate(CASES):
        rng = np.random.default_rng(100 + i)
        model = NavrosNP(cfg, seed=i, dtype=np.float64)
        # romper la simetría de las ganancias iniciales (=1) para que su gradiente se pruebe de verdad
        for kname, v in model.P.items():
            if v.ndim == 1:
                v += 0.1 * rng.standard_normal(v.shape)
        batch = make_batch(cfg, B, T, rng, pad)
        loss, rep = directional_check(model, batch, r=r, k=k, seed=i)
        worst = max(v["best_err"] for v in rep.values())
        worst_all = max(worst_all, worst)
        ok = worst < TOL
        summary.append(dict(case=name, loss=loss, worst_best_err=worst, ok=ok, n_tensors=len(rep),
                            per_tensor={t: dict(err=v["best_err"], eps=v["best_eps"], slope=v["slope"],
                                                grad_norm=v["grad_norm"]) for t, v in rep.items()}))
        print(f"\n== {name}  (r={r}, k={k}, padding={pad})  loss={loss:.6f}  → peor error {worst:.2e}  {'OK' if ok else 'FALLO'}")
        if not args.quiet:
            print(f"   {'tensor':<14}{'|∇|':>10}{'mejor err':>11}{'eps':>8}{'pend.':>7}")
            for t, v in rep.items():
                print(f"   {t:<14}{v['grad_norm']:>10.2e}{v['best_err']:>11.2e}{v['best_eps']:>8.0e}{v['slope']:>7.2f}")

    # Ilustración: la entrada de menor |gradiente| juzgada sola, con eps grande
    cfg = CASES[1][1]
    rng = np.random.default_rng(101)
    model = NavrosNP(cfg, seed=1)
    batch = make_batch(cfg, 3, 6, rng, True)
    _, G, _ = model.loss_and_grads(batch, r=5)
    g = np.abs(G["core.0.0.wq"])
    idx = np.unravel_index(np.argmin(g + (g == 0) * 1e9), g.shape)
    a, n, e = entrywise_check(model, batch, "core.0.0.wq", idx, r=5, eps=1e-2)
    print(f"\nIlustración (entrada suelta de core.0.0.wq con |g|={abs(a):.1e}, eps=1e-2): "
          f"analítico {a:.3e}, numérico {n:.3e}, error rel. {e:.1e}")

    print(f"\nRESUMEN: {len(CASES)} configuraciones, peor error (mejor eps por tensor) = {worst_all:.2e}"
          f"  [tolerancia {TOL:.0e}]  → {'OK' if worst_all < TOL else 'FALLO'}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(dict(tol=TOL, worst=worst_all, cases=summary), indent=1))
    sys.exit(0 if worst_all < TOL else 1)


if __name__ == "__main__":
    main()
