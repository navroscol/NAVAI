"""Tareas de estrés sobre el mapa de Collatz, en bits LSB-primero.

Tareas:
  paso-k   : dado n en binario (L bits), emitir T^k(n) en binario (L + 2k bits). Etiquetado por posición.
             Para k fijo es una función de estado finito (transductor con acarreo), así que un modelo
             con memoria acotada puede representarla exactamente para cualquier L.
  parada   : dado n, decidir si su tiempo total de parada supera la mediana de su longitud.
             No es de estado finito: es la frontera "Smale 18" del banco de pruebas.
"""
from __future__ import annotations

import numpy as np

PAD = 2  # símbolo de relleno en la entrada (vocabulario: 0, 1, PAD)


def T(n: int) -> int:
    return n // 2 if n % 2 == 0 else 3 * n + 1


def T_k(n: int, k: int) -> int:
    for _ in range(k):
        n = T(n)
    return n


def tiempo_parada(n: int, tope: int = 10_000) -> int:
    s = 0
    while n != 1 and s < tope:
        n = T(n)
        s += 1
    return s


def a_bits(n: int, ancho: int) -> np.ndarray:
    """n → bits LSB-primero, ancho fijo."""
    return np.array([(n >> i) & 1 for i in range(ancho)], dtype=np.int64)


def muestras_n(rng: np.random.Generator, L: int, B: int) -> np.ndarray:
    """B enteros con exactamente L bits (bit L-1 encendido)."""
    lo, hi = 1 << (L - 1), 1 << L
    return rng.integers(lo, hi, size=B, dtype=np.int64)


def lote_paso_k(rng: np.random.Generator, L: int, B: int, k: int):
    """Entrada: n en L bits + relleno hasta L+2k. Salida: T^k(n) en L+2k bits."""
    ancho = L + 2 * k
    ns = muestras_n(rng, L, B)
    x = np.full((B, ancho), PAD, dtype=np.int64)
    y = np.zeros((B, ancho), dtype=np.int64)
    for i, n in enumerate(ns):
        x[i, :L] = a_bits(int(n), L)
        y[i] = a_bits(T_k(int(n), k), ancho)
    return x, y


_medianas: dict[int, float] = {}


def mediana_parada(L: int, rng_semilla: int = 12345, N: int = 4000) -> float:
    if L not in _medianas:
        rng = np.random.default_rng(rng_semilla + L)
        ns = muestras_n(rng, L, N)
        _medianas[L] = float(np.median([tiempo_parada(int(n)) for n in ns]))
    return _medianas[L]


def lote_parada(rng: np.random.Generator, L: int, B: int):
    """Entrada: n en L bits. Salida: 1 si tiempo de parada > mediana(L)."""
    ns = muestras_n(rng, L, B)
    med = mediana_parada(L)
    x = np.stack([a_bits(int(n), L) for n in ns])
    y = np.array([int(tiempo_parada(int(n)) > med) for n in ns], dtype=np.int64)
    return x, y


def lote(tarea: str, rng: np.random.Generator, L: int, B: int):
    if tarea.startswith("paso-"):
        return lote_paso_k(rng, L, B, int(tarea.split("-")[1]))
    if tarea == "parada":
        return lote_parada(rng, L, B)
    raise ValueError(tarea)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x, y = lote_paso_k(rng, 6, 3, 1)
    for xi, yi in zip(x, y):
        n = sum(int(b) << i for i, b in enumerate(xi[:6]))
        m = sum(int(b) << i for i, b in enumerate(yi))
        assert m == T(n), (n, m)
        print(n, "→", m, xi, yi)
    x, y = lote_parada(rng, 10, 5)
    print("mediana(10) =", mediana_parada(10), "etiquetas", y)
    assert tiempo_parada(27) == 111
    print("ok")
