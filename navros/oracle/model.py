"""NAVROS en NumPy puro: forward, backward manual y muestreo de r.

    preludio:  x0 = emb[tokens] (+ abaco[pos]) → capas de preludio
    núcleo:    h ← rmsnorm_sin_parámetros( bloque_t( h + x0 ) )      r veces
    coda:      capas de coda → logits = rmsnorm(h; g_f) @ emb.T · 1/√d   (pesos atados)

bloque_t es el mismo en todas las iteraciones (recurrente) o distinto en cada una
(referencia fija, r = n_blocks). Con BPTT truncado solo las últimas k iteraciones
guardan activaciones; el estado que entra en ellas se trata como constante.
"""
from __future__ import annotations

import numpy as np

from ..config import NavrosConfig
from .layers import LAYER_KEYS, layer_bwd, layer_fwd, rmsnorm_bwd, rmsnorm_fwd, rope_tables


def layer_prefixes(cfg: NavrosConfig):
    pre = [f"pre.{i}." for i in range(cfg.n_pre)]
    core = [[f"core.{b}.{l}." for l in range(cfg.n_core)] for b in range(cfg.n_blocks if cfg.n_core else 0)]
    coda = [f"coda.{i}." for i in range(cfg.n_coda)]
    return pre, core, coda


def init_params(cfg: NavrosConfig, seed: int = 0, dtype=np.float64) -> dict:
    """emb, abaco ~ N(0,1); matrices ~ N(0, 1/entrada); ganancias = 1. Sin sesgos."""
    rng = np.random.default_rng(seed)
    P = {"emb": rng.standard_normal((cfg.vocab, cfg.d))}
    if cfg.abacus:
        P["abaco"] = rng.standard_normal((cfg.abacus, cfg.d))
    d, f = cfg.d, cfg.ffn
    shapes = {"g1": (d,), "wq": (d, d), "wk": (d, d), "wv": (d, d), "wo": (d, d),
              "g2": (d,), "w1": (f, d), "w3": (f, d), "w2": (d, f)}
    pre, core, coda = layer_prefixes(cfg)
    for p in pre + [x for blk in core for x in blk] + coda:
        for k in LAYER_KEYS:
            shp = shapes[k]
            P[p + k] = np.ones(shp) if len(shp) == 1 else rng.standard_normal(shp) / np.sqrt(shp[1])
    P["norm_f"] = np.ones(d)
    return {k: v.astype(dtype) for k, v in P.items()}


def sample_r(rng: np.random.Generator, cfg: NavrosConfig) -> int:
    """Poisson-lognormal: r = 1 + Poisson(λ), λ ~ LogNormal(log(r̄−1) − σ²/2, σ) ⇒ E[r] = r̄."""
    lam_mean = max(cfg.r_mean - 1.0, 1e-3)
    lam = np.exp(rng.normal(np.log(lam_mean) - 0.5 * cfg.r_sigma ** 2, cfg.r_sigma))
    return int(min(1 + rng.poisson(lam), cfg.r_max))


def attention_mask(B, T, causal, valid):
    """bool (B|1,1,T,T), True = permitido. valid: (B,T) bool o None (padding de claves)."""
    m = None
    if causal:
        m = np.tril(np.ones((T, T), dtype=bool))[None, None]
    if valid is not None:
        kv = valid[:, None, None, :]
        m = kv if m is None else (m & kv)
    return m


class NavrosNP:
    def __init__(self, cfg: NavrosConfig, params: dict | None = None, seed: int = 0, dtype=np.float64):
        self.cfg = cfg
        self.P = params if params is not None else init_params(cfg, seed, dtype)
        self.pre, self.core, self.coda = layer_prefixes(cfg)

    # ------------------------------------------------------------------ forward
    def _block(self, t):
        return self.core[0] if self.cfg.recurrent else self.core[t]

    def _iter(self, h, x0, t, rope, allowed, keep):
        cfg, P = self.cfg, self.P
        u = h + x0
        caches = []
        for p in self._block(t):
            u, c = layer_fwd(u, P, p, cfg.n_heads, cfg.scale_core, rope, allowed, cfg.norm_eps)
            caches.append(c)
        h, cn = rmsnorm_fwd(u, None, cfg.norm_eps)
        return h, ((caches, cn) if keep else None)

    def forward(self, batch, r=None, k=None, h_start=None, keep=True):
        """Devuelve (loss, cache). batch: tokens, targets, weights (B,T); opcionales abacus, valid.

        r: iteraciones (obligatorio si hay núcleo recurrente; la referencia fija usa n_blocks).
        k: iteraciones con gradiente (None → todas). h_start: estado que entra en la primera
        iteración con gradiente (para comprobar el BPTT truncado con diferencias finitas).
        """
        cfg, P = self.cfg, self.P
        tok = batch["tokens"]
        B, T = tok.shape
        dt = P["emb"].dtype
        x = P["emb"][tok]
        if cfg.abacus:
            x = x + P["abaco"][batch["abacus"]]
        rope = rope_tables(T, cfg.head_dim, cfg.rope_theta, dt) if cfg.rope else None
        allowed = attention_mask(B, T, cfg.causal, batch.get("valid"))

        c_pre = []
        for p in self.pre:
            x, c = layer_fwd(x, P, p, cfg.n_heads, cfg.scale_stack, rope, allowed, cfg.norm_eps)
            c_pre.append(c)
        x0 = x

        c_it, h_entry = [], None
        if cfg.n_core:
            if not cfg.recurrent:
                assert r is None or r == cfg.n_blocks, "la referencia fija siempre usa r = n_blocks"
                r = cfg.n_blocks
            assert r is not None and r >= 1
            k = r if k is None else min(k, r)
            n_ng = r - k
            if h_start is None:
                h = np.zeros_like(x0)
                for t in range(n_ng):
                    h, _ = self._iter(h, x0, t, rope, allowed, keep=False)
            else:
                h = h_start
            h_entry = h
            for t in range(n_ng, r):
                h, c = self._iter(h, x0, t, rope, allowed, keep=keep)
                c_it.append(c)
        else:
            h = x0

        c_coda = []
        for p in self.coda:
            h, c = layer_fwd(h, P, p, cfg.n_heads, cfg.scale_stack, rope, allowed, cfg.norm_eps)
            c_coda.append(c)

        f, cf = rmsnorm_fwd(h, P["norm_f"], cfg.norm_eps)
        logits = (f @ P["emb"].T) * cfg.logit_scale
        loss, dlogits = weighted_xent(logits, batch["targets"], batch["weights"])
        cache = dict(tok=tok, ab=batch.get("abacus"), c_pre=c_pre, c_it=c_it, c_coda=c_coda,
                     f=f, cf=cf, dlogits=dlogits, h_entry=h_entry, logits=logits)
        return loss, cache

    # ----------------------------------------------------------------- backward
    def backward(self, cache) -> dict:
        cfg, P = self.cfg, self.P
        G = {k: np.zeros_like(v) for k, v in P.items()}
        d = cfg.d
        dl = cache["dlogits"] * cfg.logit_scale
        f = cache["f"]
        G["emb"] += dl.reshape(-1, cfg.vocab).T @ f.reshape(-1, d)
        dh, G["norm_f"] = rmsnorm_bwd(dl @ P["emb"], cache["cf"])

        for c in reversed(cache["c_coda"]):
            dh = layer_bwd(dh, c, G)

        if cfg.n_core:
            dx0 = np.zeros_like(dh)
            for caches, cn in reversed(cache["c_it"]):
                du, _ = rmsnorm_bwd(dh, cn)
                for c in reversed(caches):
                    du = layer_bwd(du, c, G)
                dx0 += du
                dh = du
            # dh que entra en la primera iteración con gradiente se descarta (stop-gradient).
        else:
            dx0 = dh

        for c in reversed(cache["c_pre"]):
            dx0 = layer_bwd(dx0, c, G)
        np.add.at(G["emb"], cache["tok"], dx0)
        if cfg.abacus:
            np.add.at(G["abaco"], cache["ab"], dx0)
        return G

    def loss_and_grads(self, batch, r=None, k=None, h_start=None):
        loss, cache = self.forward(batch, r=r, k=k, h_start=h_start)
        return loss, self.backward(cache), cache

    def logits(self, batch, r=None):
        _, cache = self.forward(batch, r=r, keep=False)
        return cache["logits"]


def weighted_xent(logits, targets, weights):
    """Σ w·CE / Σ w. Devuelve (loss, dloss/dlogits)."""
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    s = e.sum(axis=-1, keepdims=True)
    logp = z - np.log(s)
    W = weights.sum()
    nll = -np.take_along_axis(logp, targets[..., None], axis=-1)[..., 0]
    loss = float((weights * nll).sum() / W)
    dz = e / s
    np.put_along_axis(dz, targets[..., None], np.take_along_axis(dz, targets[..., None], -1) - 1.0, -1)
    dz *= (weights / W)[..., None]
    return loss, dz
