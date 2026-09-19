"""Verificación de gradientes contra diferencias finitas.

Protocolo:
  1. Se juzga cada tensor por su derivada direccional completa ⟨∇W, U⟩ con U aleatoria
     de norma 1, no por entradas sueltas: con gradientes de magnitud ~1e-6 el error
     relativo de una entrada lo domina el redondeo y da falsos positivos.
  2. Se barre epsilon. Si el error cae como eps² en la zona de eps grande es
     truncamiento de las diferencias finitas; si se estanca lejos de 0 es un bug.
  3. Con BPTT truncado el gradiente analítico no es el de la pérdida completa, sino el
     de la pérdida con el estado de entrada congelado: las diferencias finitas se
     calculan con ese mismo estado fijo (h_start), que es lo que el código afirma computar.
"""
from __future__ import annotations

import numpy as np

EPS_SWEEP = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7)


def _rel(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


def directional_check(model, batch, r=None, k=None, eps_sweep=EPS_SWEEP, seed=0):
    loss, G, cache = model.loss_and_grads(batch, r=r, k=k)
    truncated = k is not None and r is not None and k < r
    h_start = cache["h_entry"] if truncated else None
    rng = np.random.default_rng(seed)
    report = {}
    for name, p in model.P.items():
        U = rng.standard_normal(p.shape)
        U /= np.linalg.norm(U)
        analytic = float(np.sum(G[name] * U))
        p0 = p.copy()
        rows = []
        for eps in eps_sweep:
            p[...] = p0 + eps * U
            lp, _ = model.forward(batch, r=r, k=k, h_start=h_start, keep=False)
            p[...] = p0 - eps * U
            lm, _ = model.forward(batch, r=r, k=k, h_start=h_start, keep=False)
            num = (lp - lm) / (2 * eps)
            rows.append((eps, num, _rel(analytic, num)))
        p[...] = p0
        best = min(rows, key=lambda x: x[2])
        # pendiente log-log del error entre los dos eps más grandes (≈2 si es truncamiento)
        (e1, _, r1), (e2, _, r2) = rows[0], rows[1]
        slope = np.log(max(r1, 1e-300) / max(r2, 1e-300)) / np.log(e1 / e2)
        report[name] = dict(analytic=analytic, grad_norm=float(np.linalg.norm(G[name])),
                            best_eps=best[0], best_err=best[2], slope=float(slope), rows=rows)
    return loss, report


def entrywise_check(model, batch, name, idx, r=None, k=None, eps=1e-4):
    """Comprobación de una entrada suelta; se usa solo para ilustrar por qué engaña."""
    loss, G, cache = model.loss_and_grads(batch, r=r, k=k)
    h_start = cache["h_entry"] if (k is not None and r is not None and k < r) else None
    p = model.P[name]
    v0 = p[idx]
    p[idx] = v0 + eps
    lp, _ = model.forward(batch, r=r, k=k, h_start=h_start, keep=False)
    p[idx] = v0 - eps
    lm, _ = model.forward(batch, r=r, k=k, h_start=h_start, keep=False)
    p[idx] = v0
    num = (lp - lm) / (2 * eps)
    return float(G[name][idx]), float(num), _rel(float(G[name][idx]), num)
