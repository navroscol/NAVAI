"""Suma de n dígitos, sin tokens intermedios.

Entrada (dígitos de menor a mayor peso):   a0 a1 … a(n−1) + b0 b1 … b(n−1) =
Salida en la misma pasada: en la posición de b_i el dígito s_i de la suma; en '=' el
acarreo final s_n. El modelo tiene que propagar acarreos: el error en una posición
arrastra a las siguientes, así que secuencia y posición van juntas.

Ábaco: a_i, b_i reciben el índice i+1 (mismo valor posicional), '=' recibe n+1.
En entrenamiento se suma un desplazamiento aleatorio para que se entrenen índices
mayores que la longitud vista (McLeish et al. 2024). En evaluación, desplazamiento 0.

Verificación independiente: se reconstruyen los enteros y se suman con la aritmética
de Python; no comparte nada con el algoritmo de columnas que genera las etiquetas.
"""
from __future__ import annotations

import numpy as np

from .common import Example, Pool

PLUS, EQ, PAD = 10, 11, 12
VOCAB = 13
SYMBOLS = "0123456789+=_"


def generate(rng, B, n, abacus_size, train):
    a = rng.integers(0, 10, (B, n))
    b = rng.integers(0, 10, (B, n))
    s = np.zeros((B, n + 1), dtype=np.int64)
    c = np.zeros(B, dtype=np.int64)
    for i in range(n):  # algoritmo de columnas (genera las etiquetas)
        t = a[:, i] + b[:, i] + c
        s[:, i], c = t % 10, t // 10
    s[:, n] = c
    T = 2 * n + 2
    tokens = np.concatenate([a, np.full((B, 1), PLUS), b, np.full((B, 1), EQ)], axis=1)
    targets = np.zeros((B, T), dtype=np.int64)
    targets[:, n + 1:] = s
    weights = np.zeros((B, T))
    weights[:, n + 1:] = 1.0
    pos = np.concatenate([np.arange(1, n + 1), [0], np.arange(1, n + 1), [n + 1]])
    max_off = abacus_size - (n + 2)
    assert max_off >= 0, f"ábaco de {abacus_size} demasiado pequeño para n={n}"
    off = rng.integers(0, max_off + 1, (B, 1)) if train else np.zeros((B, 1), dtype=np.int64)
    abacus = np.where(pos[None] > 0, pos[None] + off, 0)
    return tokens, targets, weights, abacus


def verify(tokens, targets, n):
    """Vía independiente: enteros de Python."""
    for row_t, row_y in zip(tokens, targets):
        A = int("".join(str(d) for d in row_t[:n][::-1]))
        B = int("".join(str(d) for d in row_t[n + 1:2 * n + 1][::-1]))
        S = int("".join(str(d) for d in row_y[n + 1:][::-1]))
        if A + B != S:
            raise AssertionError(f"etiqueta errónea: {A}+{B}≠{S}")


def build_pool(rng, lengths, per_len, abacus_size, train=True) -> Pool:
    by_len = {}
    for n in lengths:
        tok, tgt, w, ab = generate(rng, per_len, n, abacus_size, train)
        verify(tok, tgt, n)
        by_len[n] = [Example(tok[i], tgt[i], w[i], ab[i]) for i in range(per_len)]
    return Pool(by_len, PAD)


def render(tokens) -> str:
    return "".join(SYMBOLS[t] for t in tokens)
