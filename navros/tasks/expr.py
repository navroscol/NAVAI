"""Evaluación de expresiones módulo 10, supervisando cada subexpresión.

Ejemplo:  3+4*(2-7)=
Etiquetas (valor mod 10 de la subexpresión que cierra cada posición):
  * cada operador → valor del subárbol cuya raíz es ese operador
  * cada ')'      → valor de lo que encierra
  * '='           → valor total
Los dígitos y '(' no se supervisan. Una sola etiqueta por secuencia es supervisión
demasiado escasa: el modelo apenas aprende. Por eso se supervisa cada subexpresión.

Longitud = número de operadores. Paréntesis mínimos según precedencia y asociatividad
izquierda de Python (el hijo derecho de igual precedencia siempre lleva paréntesis).

Verificación independiente: se parsea la cadena con `ast` de Python y cada etiqueta se
recalcula con eval() sobre el nodo que Python asigna a esa posición. Si la gramática o
la precedencia del generador discrepan de las de Python (p. ej. un paréntesis omitido
que cambia la asociatividad), salta aunque el valor total coincida.
"""
from __future__ import annotations

import ast

import numpy as np

from .common import Example, Pool, random_positions

SYMBOLS = "0123456789+-*()=_"
TOK = {c: i for i, c in enumerate(SYMBOLS)}
PAD = TOK["_"]
VOCAB = len(SYMBOLS)
PREC = {"+": 1, "-": 1, "*": 2}
OPS = ("+", "-", "*")


def gen_tree(rng, n):
    if n == 0:
        return int(rng.integers(0, 10))
    i = int(rng.integers(0, n))
    return (OPS[int(rng.integers(0, 3))], gen_tree(rng, i), gen_tree(rng, n - 1 - i))


def value(node):
    """Evaluador propio, módulo 10 en cada paso (genera las etiquetas)."""
    if isinstance(node, int):
        return node
    op, l, r = node
    a, b = value(l), value(r)
    return (a + b) % 10 if op == "+" else (a - b) % 10 if op == "-" else (a * b) % 10


def render(node, chars, labels):
    if isinstance(node, int):
        chars.append(str(node))
        labels.append(-1)
        return
    op, l, r = node
    for child, side in ((l, "L"), (r, "R")):
        if side == "R":
            chars.append(op)
            labels.append(value(node))
        paren = not isinstance(child, int) and (
            PREC[child[0]] < PREC[op] if side == "L" else PREC[child[0]] <= PREC[op])
        if paren:
            chars.append("(")
            labels.append(-1)
        render(child, chars, labels)
        if paren:
            chars.append(")")
            labels.append(value(child))


def python_labels(s: str) -> dict[int, int]:
    """Etiquetas recalculadas solo con el parser y eval() de Python."""
    tree = ast.parse(s, mode="eval")
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp):
            p = node.left.end_col_offset
            while s[p] == ")":
                p += 1
            out[p] = eval(compile(ast.Expression(body=node), "<e>", "eval")) % 10
    stack = []
    for i, c in enumerate(s):
        if c == "(":
            stack.append(i)
        elif c == ")":
            j = stack.pop()
            out[i] = eval(s[j + 1:i]) % 10
    out[len(s)] = eval(s) % 10  # posición de '='
    return out


def make_example(rng, n_ops, verify=True) -> Example:
    tree = gen_tree(rng, n_ops)
    chars, labels = [], []
    render(tree, chars, labels)
    s = "".join(chars)
    chars.append("=")
    labels.append(value(tree))
    if verify:
        mine = {i: v for i, v in enumerate(labels) if v >= 0}
        if mine != python_labels(s):
            raise AssertionError(f"etiquetas discrepan de Python en {s!r}")
    tokens = np.array([TOK[c] for c in chars], dtype=np.int64)
    lab = np.array(labels, dtype=np.int64)
    return Example(tokens, np.maximum(lab, 0), (lab >= 0).astype(np.float64), None)


def build_pool(rng, lengths, per_len, verify=True, abacus_size=0, train=True, rope_range=0) -> Pool:
    """abacus_size > 0: cada token recibe su posición 1..T como índice de ábaco, más un
    desplazamiento aleatorio en entrenamiento (0 en evaluación). Es la alternativa a RoPE
    para que índices mayores que la longitud entrenada tengan embedding entrenado."""
    by_len = {}
    for n in lengths:
        exs = [make_example(rng, n, verify) for _ in range(per_len)]
        if abacus_size:
            for e in exs:
                T = len(e.tokens)
                assert T + 1 <= abacus_size, f"ábaco de {abacus_size} demasiado pequeño para T={T}"
                off = int(rng.integers(0, abacus_size - T)) if train else 0
                e.abacus = np.arange(1, T + 1) + off
        if rope_range:
            for e in exs:
                e.pos = random_positions(rng, len(e.tokens), rope_range)
        by_len[n] = exs
    return Pool(by_len, PAD)


def to_str(tokens) -> str:
    return "".join(SYMBOLS[t] for t in tokens if t != PAD)
