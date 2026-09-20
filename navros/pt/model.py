"""NAVROS en PyTorch. Mismo grafo y mismos nombres de parámetros que navros/oracle.

named_parameters() produce exactamente las claves del oráculo ("emb", "core.0.0.wq",
"norm_f", ...), así que cargar pesos NumPy es copiar un dict.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import NavrosConfig


def rmsnorm(x, g, eps):
    """En fp16/bf16 se calcula en float32 (x² desborda fp16 por encima de ~256)."""
    xf = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    y = y.to(x.dtype)
    return y * g if g is not None else y


def rope_tables(T, head_dim, theta, device, dtype, offset=0):
    half = head_dim // 2
    freqs = theta ** (-torch.arange(half, dtype=torch.float64) / half)
    ang = torch.outer(torch.arange(offset, offset + T, dtype=torch.float64), freqs)
    return ang.cos().to(device, dtype), ang.sin().to(device, dtype)


def rope_tables_pos(pos, head_dim, theta, dtype):
    """pos: (B,T) enteros. Devuelve cos, sin de forma (B,1,T,hd/2)."""
    half = head_dim // 2
    freqs = (theta ** (-torch.arange(half, dtype=torch.float64) / half)).to(pos.device)
    ang = pos[..., None].double() * freqs
    return ang.cos()[:, None].to(dtype), ang.sin()[:, None].to(dtype)


def apply_rope(x, cos, sin):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Layer(nn.Module):
    """Pre-norma, sin sesgos: y = x + s·Attn(norm(x)); z = y + s·SwiGLU(norm(y))."""

    def __init__(self, cfg: NavrosConfig, scale: float):
        super().__init__()
        d, f = cfg.d, cfg.ffn
        self.cfg, self.scale = cfg, scale
        self.g1 = nn.Parameter(torch.ones(d))
        self.wq = nn.Parameter(torch.empty(d, d))
        self.wk = nn.Parameter(torch.empty(d, d))
        self.wv = nn.Parameter(torch.empty(d, d))
        self.wo = nn.Parameter(torch.empty(d, d))
        self.g2 = nn.Parameter(torch.ones(d))
        self.w1 = nn.Parameter(torch.empty(f, d))
        self.w3 = nn.Parameter(torch.empty(f, d))
        self.w2 = nn.Parameter(torch.empty(d, f))

    def attn(self, u, rope, mask, manual):
        B, T, d = u.shape
        H = self.cfg.n_heads
        split = lambda z: z.view(B, T, H, d // H).transpose(1, 2)
        q, k, v = split(u @ self.wq.T), split(u @ self.wk.T), split(u @ self.wv.T)
        if rope is not None:
            q, k = apply_rope(q, *rope), apply_rope(k, *rope)
        causal = isinstance(mask, str)  # "causal": máscara triangular implícita (kernels eficientes)
        if manual:
            s = (q @ k.transpose(-1, -2)) / math.sqrt(d // H)
            if causal:
                s = s.masked_fill(~torch.ones(T, T, dtype=torch.bool, device=u.device).tril(), float("-inf"))
            elif mask is not None:
                s = s.masked_fill(~mask, float("-inf"))
            o = torch.softmax(s, dim=-1) @ v
        elif causal:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return o.transpose(1, 2).reshape(B, T, d) @ self.wo.T

    def forward(self, x, rope, mask, manual=False):
        eps = self.cfg.norm_eps
        x = x + self.scale * self.attn(rmsnorm(x, self.g1, eps), rope, mask, manual)
        u = rmsnorm(x, self.g2, eps)
        return x + self.scale * ((F.silu(u @ self.w1.T) * (u @ self.w3.T)) @ self.w2.T)


class Navros(nn.Module):
    def __init__(self, cfg: NavrosConfig):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Parameter(torch.empty(cfg.vocab, cfg.d))
        if cfg.abacus:
            self.abaco = nn.Parameter(torch.empty(cfg.abacus, cfg.d))
        self.pre = nn.ModuleList(Layer(cfg, cfg.scale_stack) for _ in range(cfg.n_pre))
        nb = cfg.n_blocks if cfg.n_core else 0
        self.core = nn.ModuleList(nn.ModuleList(Layer(cfg, cfg.scale_core) for _ in range(cfg.n_core)) for _ in range(nb))
        self.coda = nn.ModuleList(Layer(cfg, cfg.scale_stack) for _ in range(cfg.n_coda))
        self.norm_f = nn.Parameter(torch.ones(cfg.d))
        self.manual_attn = False
        self.ckpt_fn = None   # p. ej. torch.utils.checkpoint.checkpoint: recomputa cada capa en el backward
        self.reset_parameters()

    def _layer(self, layer, x, rope, mask):
        if self.ckpt_fn is not None and torch.is_grad_enabled():
            return self.ckpt_fn(layer, x, rope, mask, self.manual_attn)
        return layer(x, rope, mask, self.manual_attn)

    @torch.no_grad()
    def reset_parameters(self, generator=None):
        for name, p in self.named_parameters():
            if name in ("emb", "abaco"):
                p.normal_(0.0, 1.0, generator=generator)
            elif p.ndim == 1:
                p.fill_(1.0)
            else:
                p.normal_(0.0, 1.0 / math.sqrt(p.shape[1]), generator=generator)

    @torch.no_grad()
    def load_numpy(self, P: dict):
        own = dict(self.named_parameters())
        assert set(own) == set(P), (set(own) ^ set(P))
        for k, v in P.items():
            own[k].copy_(torch.from_numpy(np.asarray(v)))

    # ------------------------------------------------------------------ forward
    def _mask(self, B, T, valid, device):
        if self.cfg.causal and valid is None:
            return "causal"
        m = None
        if self.cfg.causal:
            m = torch.ones(T, T, dtype=torch.bool, device=device).tril()[None, None]
        if valid is not None:
            kv = valid[:, None, None, :]
            m = kv if m is None else (m & kv)
        return m

    def _iter(self, h, x0, t, rope, mask):
        u = h + x0
        for layer in (self.core[0] if self.cfg.recurrent else self.core[t]):
            u = self._layer(layer, u, rope, mask)
        return rmsnorm(u, None, self.cfg.norm_eps)

    def prelude(self, tokens, abacus=None, valid=None, pos=None):
        cfg = self.cfg
        B, T = tokens.shape
        x = self.emb[tokens]
        if cfg.abacus:
            x = x + self.abaco[abacus]
        if not cfg.rope:
            rope = None
        elif pos is not None:
            rope = rope_tables_pos(pos, cfg.head_dim, cfg.rope_theta, x.dtype)
        else:
            rope = rope_tables(T, cfg.head_dim, cfg.rope_theta, x.device, x.dtype)
        mask = self._mask(B, T, valid, x.device)
        for layer in self.pre:
            x = self._layer(layer, x, rope, mask)
        return x, rope, mask

    def coda_logits(self, h, rope, mask):
        for layer in self.coda:
            h = self._layer(layer, h, rope, mask)
        return (rmsnorm(h, self.norm_f, self.cfg.norm_eps) @ self.emb.T) * self.cfg.logit_scale

    def forward(self, tokens, abacus=None, valid=None, r=None, k=None, h_start=None, return_state=False, pos=None):
        """Devuelve logits (B,T,V). Las primeras r−k iteraciones corren sin gradiente."""
        cfg = self.cfg
        x0, rope, mask = self.prelude(tokens, abacus, valid, pos)
        h_entry = None
        if cfg.n_core:
            if not cfg.recurrent:
                assert r is None or r == cfg.n_blocks
                r = cfg.n_blocks
            k = r if k is None else min(k, r)
            n_ng = r - k
            if h_start is None:
                h = torch.zeros_like(x0)
                if n_ng:
                    with torch.no_grad():
                        x0d = x0.detach()
                        for t in range(n_ng):
                            h = self._iter(h, x0d, t, rope, mask)
            else:
                h = h_start
            h_entry = h
            for t in range(n_ng, r):
                h = self._iter(h, x0, t, rope, mask)
        else:
            h = x0
        logits = self.coda_logits(h, rope, mask)
        return (logits, h_entry) if return_state else logits

    @torch.no_grad()
    def iterate(self, tokens, abacus=None, valid=None, r_max=64, tol=None):
        """Inferencia adaptativa: itera el núcleo y devuelve los logits tras cada iteración.

        Si tol está definido, un ejemplo se congela cuando el cambio relativo del estado
        ‖h_t − h_{t−1}‖/‖h_t‖ baja de tol (salida por convergencia). Devuelve
        (lista de logits por iteración, iteraciones usadas por ejemplo).
        """
        cfg = self.cfg
        assert cfg.recurrent
        x0, rope, mask = self.prelude(tokens, abacus, valid)
        h = torch.zeros_like(x0)
        B = tokens.shape[0]
        done = torch.zeros(B, dtype=torch.bool, device=x0.device)
        used = torch.zeros(B, dtype=torch.long, device=x0.device)
        outs = []
        for t in range(r_max):
            hn = self._iter(h, x0, 0, rope, mask)
            if tol is not None:
                delta = (hn - h).flatten(1).norm(dim=1) / hn.flatten(1).norm(dim=1)
                hn = torch.where(done[:, None, None], h, hn)
                used += (~done).long()
                done = done | (delta < tol)
            else:
                used += 1
            h = hn
            outs.append(self.coda_logits(h, rope, mask))
            if tol is not None and bool(done.all()):
                break
        return outs, used


def weighted_xent(logits, targets, weights):
    nll = F.cross_entropy(logits.flatten(0, -2), targets.flatten(), reduction="none")
    w = weights.flatten()
    return (nll * w).sum() / w.sum()


def sample_r(gen: np.random.Generator, cfg: NavrosConfig) -> int:
    from ..oracle.model import sample_r as _s
    return _s(gen, cfg)
