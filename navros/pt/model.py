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


def _rope_freqs(rotary_dim, theta, factors=None):
    half = rotary_dim // 2
    freqs = theta ** (-torch.arange(half, dtype=torch.float64) / half)
    if factors:
        freqs = freqs / torch.tensor(factors, dtype=torch.float64)   # LongRoPE: divisor por frecuencia
    return freqs


def rope_tables(T, head_dim, theta, device, dtype, offset=0, rotary_dim=None, factors=None, mscale=1.0):
    freqs = _rope_freqs(rotary_dim or head_dim, theta, factors)
    ang = torch.outer(torch.arange(offset, offset + T, dtype=torch.float64), freqs)
    return (ang.cos() * mscale).to(device, dtype), (ang.sin() * mscale).to(device, dtype)


def rope_tables_pos(pos, head_dim, theta, dtype, rotary_dim=None, factors=None, mscale=1.0):
    """pos: (B,T) enteros. Devuelve cos, sin de forma (B,1,T,rd/2)."""
    freqs = _rope_freqs(rotary_dim or head_dim, theta, factors).to(pos.device)
    ang = pos[..., None].double() * freqs
    return (ang.cos() * mscale)[:, None].to(dtype), (ang.sin() * mscale)[:, None].to(dtype)


def rope_from_cfg(cfg, T, device, dtype, offset=0, pos=None):
    """Tablas RoPE con todos los ajustes de la configuración (fracción rotada, LongRoPE)."""
    extra = dict(rotary_dim=cfg.rotary_dim, factors=cfg.rope_factors or None, mscale=cfg.rope_mscale)
    if pos is not None:
        return rope_tables_pos(pos, cfg.head_dim, cfg.rope_theta, dtype, **extra)
    return rope_tables(T, cfg.head_dim, cfg.rope_theta, device, dtype, offset=offset, **extra)


def apply_rope(x, cos, sin):
    """Rota las primeras 2·len(cos) dimensiones de cada cabeza; el resto (si lo hay) pasa sin rotar."""
    rd = 2 * cos.shape[-1]
    xr, resto = x[..., :rd], x[..., rd:]
    half = rd // 2
    x1, x2 = xr[..., :half], xr[..., half:]
    out = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out if resto.shape[-1] == 0 else torch.cat([out, resto], dim=-1)


def rasgos_rejilla(x, kind="elu"):
    """Mapa de rasgos positivo de la atención lineal. Escala 1/hd^¼ en q y k para que φ(q)·φ(k)
    tenga el mismo orden que q·k/√hd de la softmax."""
    x = x / x.shape[-1] ** 0.25
    return F.elu(x) + 1 if kind == "elu" else F.relu(x)


def atencion_rejilla(q, k, v, kind="elu", S=None, z=None, eps=1e-6):
    """Atención lineal causal con estado matricial ("rejilla"): S_t = S_{t-1} + φ(k_t) v_tᵀ, z_t = z_{t-1} + φ(k_t),
    o_t = φ(q_t) S_t / (φ(q_t)·z_t). Forma cuadrática enmascarada dentro del bloque (exacta, O(T²) como la
    softmax) más la aportación del estado previo (S, z) si lo hay. Devuelve o y el estado actualizado."""
    qf, kf = rasgos_rejilla(q, kind), rasgos_rejilla(k, kind)          # (B,H,T,hd)
    T = q.shape[2]
    A = qf @ kf.transpose(-1, -2)                                        # (B,H,T,T)
    A = A.masked_fill(~torch.ones(T, T, dtype=torch.bool, device=q.device).tril(), 0.0)
    num = A @ v                                                          # (B,H,T,hd)
    den = A.sum(-1, keepdim=True)                                        # (B,H,T,1)
    if S is not None:
        num = num + qf @ S                                               # φ(q_t) S_prev
        den = den + (qf * z[:, :, None, :]).sum(-1, keepdim=True)        # φ(q_t)·z_prev
    o = num / (den + eps)
    S_new = (kf.transpose(-1, -2) @ v) + (S if S is not None else 0)     # (B,H,hd,hd)
    z_new = kf.sum(2) + (z if z is not None else 0)                      # (B,H,hd)
    return o, S_new, z_new


class Layer(nn.Module):
    """Pre-norma, sin sesgos: y = x + s·Attn(norm(x)); z = y + s·SwiGLU(norm(y)).
    Con `grid=True` la atención es la rejilla (lineal con estado matricial) en vez de softmax."""

    def __init__(self, cfg: NavrosConfig, scale: float, grid: bool = False):
        super().__init__()
        d, f = cfg.d, cfg.ffn
        self.cfg, self.scale, self.grid = cfg, scale, grid
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
        if self.grid:
            assert causal or mask is None, "la rejilla es causal; no admite máscaras arbitrarias"
            o, _, _ = atencion_rejilla(q, k, v, self.cfg.grid_feature)
        elif manual:
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
        self.pre = nn.ModuleList(Layer(cfg, cfg.scale_stack, cfg.es_rejilla(i)) for i in range(cfg.n_pre))
        nb = cfg.n_blocks if cfg.n_core else 0
        assert not (cfg.grid_every and cfg.n_core), "la rejilla solo está implementada en la pila fija (preludio y coda)"
        self.core = nn.ModuleList(nn.ModuleList(Layer(cfg, cfg.scale_core) for _ in range(cfg.n_core)) for _ in range(nb))
        self.coda = nn.ModuleList(Layer(cfg, cfg.scale_stack, cfg.es_rejilla(cfg.n_pre + j)) for j in range(cfg.n_coda))
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
        else:
            rope = rope_from_cfg(cfg, T, x.device, x.dtype, pos=pos)
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
