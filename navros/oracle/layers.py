"""Primitivas del oráculo: forward y backward escritos a mano.

Convención de pesos: (salida, entrada), como nn.Linear, y se aplica y = x @ W.T.
Así el port PyTorch copia los tensores sin transponer nada.
Cada *_fwd devuelve (salida, cache); cada *_bwd recibe (d_salida, cache).
"""
from __future__ import annotations

import numpy as np


# --- RMSNorm (con o sin ganancia) ------------------------------------------------

def rmsnorm_fwd(x, g, eps):
    ms = np.mean(x * x, axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(ms + eps)
    xhat = x * inv
    y = xhat * g if g is not None else xhat
    return y, (xhat, inv, g)


def rmsnorm_bwd(dy, cache):
    xhat, inv, g = cache
    if g is not None:
        dg = np.sum(dy * xhat, axis=tuple(range(dy.ndim - 1)))
        dxhat = dy * g
    else:
        dg, dxhat = None, dy
    # y = x·s, s = (mean(x²)+eps)^-½  ⇒  dx = s·(dxhat − xhat·mean(dxhat·xhat))
    dx = inv * (dxhat - xhat * np.mean(dxhat * xhat, axis=-1, keepdims=True))
    return dx, dg


# --- RoPE ------------------------------------------------------------------------

def rope_tables(T, head_dim, theta, dtype):
    half = head_dim // 2
    freqs = theta ** (-np.arange(half, dtype=np.float64) / half)
    ang = np.outer(np.arange(T, dtype=np.float64), freqs)
    return np.cos(ang).astype(dtype), np.sin(ang).astype(dtype)


def rope_tables_pos(pos, head_dim, theta, dtype):
    """pos: (B,T) enteros arbitrarios (p. ej. subconjunto ordenado al azar). Devuelve (B,1,T,hd/2)."""
    half = head_dim // 2
    freqs = theta ** (-np.arange(half, dtype=np.float64) / half)
    ang = pos[..., None].astype(np.float64) * freqs
    return np.cos(ang)[:, None].astype(dtype), np.sin(ang)[:, None].astype(dtype)


def rope_fwd(x, cos, sin):
    """x: (B,H,T,hd). Rotación por mitades (convención Llama)."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


def rope_bwd(dy, cos, sin):
    """Transpuesta de la rotación (es ortogonal)."""
    half = dy.shape[-1] // 2
    d1, d2 = dy[..., :half], dy[..., half:]
    return np.concatenate([d1 * cos + d2 * sin, -d1 * sin + d2 * cos], axis=-1)


# --- Atención multi-cabeza -------------------------------------------------------

def _split(z, H):
    B, T, d = z.shape
    return z.reshape(B, T, H, d // H).transpose(0, 2, 1, 3)


def _merge(z):
    B, H, T, hd = z.shape
    return z.transpose(0, 2, 1, 3).reshape(B, T, H * hd)


def attn_fwd(u, wq, wk, wv, wo, H, rope, allowed):
    """u: (B,T,d). rope: (cos, sin) o None. allowed: bool (B|1,1,T,T) o None."""
    q, k, v = _split(u @ wq.T, H), _split(u @ wk.T, H), _split(u @ wv.T, H)
    if rope is not None:
        q, k = rope_fwd(q, *rope), rope_fwd(k, *rope)
    scale = 1.0 / np.sqrt(q.shape[-1])
    s = (q @ k.transpose(0, 1, 3, 2)) * scale
    if allowed is not None:
        s = np.where(allowed, s, -np.inf)
    s = s - s.max(axis=-1, keepdims=True)
    e = np.exp(s)
    p = e / e.sum(axis=-1, keepdims=True)
    oc = _merge(p @ v)
    out = oc @ wo.T
    return out, (u, q, k, v, p, oc, wq, wk, wv, wo, H, rope, scale)


def attn_bwd(dout, cache):
    u, q, k, v, p, oc, wq, wk, wv, wo, H, rope, scale = cache
    d = u.shape[-1]
    dwo = dout.reshape(-1, d).T @ oc.reshape(-1, d)
    do = _split(dout @ wo, H)
    dp = do @ v.transpose(0, 1, 3, 2)
    dv = p.transpose(0, 1, 3, 2) @ do
    ds = p * (dp - np.sum(dp * p, axis=-1, keepdims=True)) * scale
    dq = ds @ k
    dk = ds.transpose(0, 1, 3, 2) @ q
    if rope is not None:
        dq, dk = rope_bwd(dq, *rope), rope_bwd(dk, *rope)
    dq, dk, dv = _merge(dq), _merge(dk), _merge(dv)
    u2 = u.reshape(-1, d)
    grads = {
        "wq": dq.reshape(-1, d).T @ u2,
        "wk": dk.reshape(-1, d).T @ u2,
        "wv": dv.reshape(-1, d).T @ u2,
        "wo": dwo,
    }
    du = dq @ wq + dk @ wk + dv @ wv
    return du, grads


# --- SwiGLU ----------------------------------------------------------------------

def _sigmoid(a):
    return 0.5 * (1.0 + np.tanh(0.5 * a))  # estable para |a| grande


def swiglu_fwd(u, w1, w3, w2):
    a = u @ w1.T
    b = u @ w3.T
    sg = _sigmoid(a)
    sa = a * sg
    m = sa * b
    return m @ w2.T, (u, a, b, sg, sa, m, w1, w3, w2)


def swiglu_bwd(dout, cache):
    u, a, b, sg, sa, m, w1, w3, w2 = cache
    d, f = u.shape[-1], a.shape[-1]
    dm = dout @ w2
    da = dm * b * (sg * (1.0 + a * (1.0 - sg)))
    db = dm * sa
    u2 = u.reshape(-1, d)
    grads = {
        "w2": dout.reshape(-1, d).T @ m.reshape(-1, f),
        "w1": da.reshape(-1, f).T @ u2,
        "w3": db.reshape(-1, f).T @ u2,
    }
    return da @ w1 + db @ w3, grads


# --- Capa residual pre-norma -----------------------------------------------------
#   y = x + s·Attn(RMSNorm(x; g1))
#   z = y + s·SwiGLU(RMSNorm(y; g2))

LAYER_KEYS = ("g1", "wq", "wk", "wv", "wo", "g2", "w1", "w3", "w2")


def layer_fwd(x, P, pre, H, scale, rope, allowed, eps):
    n1, c1 = rmsnorm_fwd(x, P[pre + "g1"], eps)
    a, ca = attn_fwd(n1, P[pre + "wq"], P[pre + "wk"], P[pre + "wv"], P[pre + "wo"], H, rope, allowed)
    y = x + scale * a
    n2, c2 = rmsnorm_fwd(y, P[pre + "g2"], eps)
    m, cm = swiglu_fwd(n2, P[pre + "w1"], P[pre + "w3"], P[pre + "w2"])
    z = y + scale * m
    return z, (c1, ca, c2, cm, scale, pre)


def layer_bwd(dz, cache, G):
    """Acumula (+=) en G: los pesos del núcleo se reutilizan en cada iteración."""
    c1, ca, c2, cm, scale, pre = cache
    dn2, gm = swiglu_bwd(scale * dz, cm)
    dy_n, dg2 = rmsnorm_bwd(dn2, c2)
    dy = dz + dy_n
    dn1, ga = attn_bwd(scale * dy, ca)
    dx_n, dg1 = rmsnorm_bwd(dn1, c1)
    dx = dy + dx_n
    for name, g in (("g1", dg1), ("g2", dg2), *ga.items(), *gm.items()):
        G[pre + name] += g
    return dx
