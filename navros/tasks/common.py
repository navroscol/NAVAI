"""Pool pregenerado y métricas comunes.

Regla del proyecto: si una etiqueta no se puede comprobar por una vía independiente,
la tarea no sirve. Cada generador trae su verificador y el pool se verifica entero al
crearse; si una sola etiqueta discrepa se aborta.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Example:
    tokens: np.ndarray    # (T,)
    targets: np.ndarray   # (T,)  etiqueta (0 donde no se supervisa)
    weights: np.ndarray   # (T,)  1 donde se supervisa
    abacus: np.ndarray | None = None
    pos: np.ndarray | None = None      # posiciones de RoPE (None = 0..T−1)


def collate(examples: list[Example], pad: int) -> dict:
    T = max(len(e.tokens) for e in examples)
    B = len(examples)
    tok = np.full((B, T), pad, dtype=np.int64)
    tgt = np.zeros((B, T), dtype=np.int64)
    w = np.zeros((B, T), dtype=np.float64)
    valid = np.zeros((B, T), dtype=bool)
    extra = {f: np.zeros((B, T), dtype=np.int64) for f in ("abacus", "pos") if getattr(examples[0], f) is not None}
    for i, e in enumerate(examples):
        n = len(e.tokens)
        tok[i, :n], tgt[i, :n], w[i, :n], valid[i, :n] = e.tokens, e.targets, e.weights, True
        for f, arr in extra.items():
            arr[i, :n] = getattr(e, f)
    return dict(tokens=tok, targets=tgt, weights=w, valid=valid) | extra


def random_positions(rng, T, rope_range):
    """Posiciones aleatorizadas (Ruoss et al., 2023): subconjunto ordenado de [0, rope_range)."""
    assert T <= rope_range, f"rope_range {rope_range} < T {T}"
    return np.sort(rng.choice(rope_range, size=T, replace=False))


class Pool:
    """Ejemplos pregenerados y verificados, apilados por longitud. Muestrear un lote es
    indexar arrays: la GPU no espera a Python generando datos."""

    def __init__(self, by_len: dict[int, list[Example]], pad: int):
        self.pad = pad
        self.lengths = sorted(by_len)
        self.data = {n: collate(exs, pad) | {"_len": np.array([len(e.tokens) for e in exs])}
                     for n, exs in by_len.items()}

    def size(self, n):
        return len(self.data[n]["tokens"])

    def get(self, n, idx):
        d = self.data[n]
        T = int(d["_len"][idx].max())
        out = {k: v[idx, :T] for k, v in d.items() if k != "_len"}
        if "valid" in out and out["valid"].all():
            out.pop("valid")
        return out

    def sample(self, rng, B, length=None):
        n = int(rng.choice(self.lengths)) if length is None else length
        return self.get(n, rng.integers(0, self.size(n), B)), n

    def iter_batches(self, n, B):
        N = self.size(n)
        for i in range(0, N, B):
            yield self.get(n, np.arange(i, min(i + B, N)))

    def keys(self, n=None):
        ns = self.lengths if n is None else [n]
        return {self.data[m]["tokens"][i, :self.data[m]["_len"][i]].tobytes() for m in ns for i in range(self.size(m))}

    def drop(self, n, bad_keys):
        """Quita de la longitud n los ejemplos cuya secuencia está en bad_keys (descontaminar)."""
        d = self.data[n]
        keep = np.array([d["tokens"][i, :d["_len"][i]].tobytes() not in bad_keys for i in range(self.size(n))])
        self.data[n] = {k: v[keep] for k, v in d.items()}
        return int((~keep).sum())


def metrics(logits: np.ndarray, batch: dict) -> dict:
    """Dos métricas: secuencia completa y posición a posición (argmax, solo posiciones supervisadas)."""
    pred = logits.argmax(-1)
    sup = batch["weights"] > 0
    correct = (pred == batch["targets"]) | ~sup
    per_pos = float(((pred == batch["targets"]) & sup).sum() / sup.sum())
    per_seq = float(correct.all(axis=1).mean())
    return dict(pos=per_pos, seq=per_seq)
