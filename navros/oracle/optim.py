"""Muon y AdamW en NumPy. Mismas fórmulas, mismo orden de operaciones que navros/pt/optim.py.

Muon (Keller Jordan, 2024): momento → Nesterov → ortogonalización por Newton-Schulz
quíntico → escala √max(1, filas/columnas). Solo para matrices entre espacios de
features (wq, wk, wv, wo, w1, w2, w3). emb, abaco y ganancias de norma van por AdamW.
"""
from __future__ import annotations

import numpy as np

NS_COEFS = (3.4445, -4.7750, 2.0315)


def is_muon_param(name: str, arr: np.ndarray) -> bool:
    return arr.ndim == 2 and name not in ("emb", "abaco")


def newton_schulz5(G, steps=5, eps=1e-7):
    a, b, c = NS_COEFS
    X = G
    tr = G.shape[0] > G.shape[1]
    if tr:
        X = X.T
    X = X / (np.linalg.norm(X) + eps)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X.T if tr else X


class MuonNP:
    """Híbrido: Muon para matrices ocultas, AdamW para el resto."""

    def __init__(self, params: dict, lr_muon=0.02, lr_adam=3e-3, momentum=0.95, nesterov=True,
                 ns_steps=5, wd_muon=0.0, wd_adam=0.0, betas=(0.9, 0.95), eps=1e-8, use_muon=True):
        self.P = params
        self.lr_muon, self.lr_adam = lr_muon, lr_adam
        self.momentum, self.nesterov, self.ns_steps = momentum, nesterov, ns_steps
        self.wd_muon, self.wd_adam = wd_muon, wd_adam
        self.b1, self.b2 = betas
        self.eps = eps
        self.muon_keys = [k for k, v in params.items() if use_muon and is_muon_param(k, v)]
        self.adam_keys = [k for k in params if k not in self.muon_keys]
        self.buf = {k: np.zeros_like(params[k]) for k in self.muon_keys}
        self.m = {k: np.zeros_like(params[k]) for k in self.adam_keys}
        self.v = {k: np.zeros_like(params[k]) for k in self.adam_keys}
        self.t = 0

    def step(self, G: dict, lr_mult: float = 1.0):
        self.t += 1
        lr = self.lr_muon * lr_mult
        for k in self.muon_keys:
            g, buf = G[k], self.buf[k]
            buf += (1.0 - self.momentum) * (g - buf)               # buf.lerp_(g, 1-β)
            u = g + self.momentum * (buf - g) if self.nesterov else buf  # g.lerp_(buf, β)
            O = newton_schulz5(u, self.ns_steps)
            O = O * max(1.0, u.shape[0] / u.shape[1]) ** 0.5
            p = self.P[k]
            p *= 1.0 - lr * self.wd_muon
            p -= lr * O
        lr = self.lr_adam * lr_mult
        bc1, bc2 = 1.0 - self.b1 ** self.t, 1.0 - self.b2 ** self.t
        for k in self.adam_keys:
            g, m, v, p = G[k], self.m[k], self.v[k], self.P[k]
            m += (1.0 - self.b1) * (g - m)
            v *= self.b2
            v += (1.0 - self.b2) * g * g
            if p.ndim == 2:
                p *= 1.0 - lr * self.wd_adam
            p -= (lr / bc1) * m / (np.sqrt(v) / bc2 ** 0.5 + self.eps)


def AdamWNP(params: dict, lr=3e-3, wd=0.0, betas=(0.9, 0.95), eps=1e-8):
    """AdamW puro = el híbrido sin parámetros Muon."""
    return MuonNP(params, lr_adam=lr, wd_adam=wd, betas=betas, eps=eps, use_muon=False)


def clip_grads(G: dict, max_norm: float) -> float:
    total = float(np.sqrt(sum(float(np.sum(g * g)) for g in G.values())))
    if max_norm:
        s = min(max_norm / (total + 1e-6), 1.0)
        for g in G.values():
            g *= s
    return total
