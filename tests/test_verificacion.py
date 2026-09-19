"""Los verificadores tienen que poder fallar: cada prueba positiva va con su control negativo."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navros.config import NavrosConfig  # noqa: E402
from navros.oracle import layers as L  # noqa: E402
from navros.oracle.gradcheck import directional_check  # noqa: E402
from navros.oracle.model import NavrosNP  # noqa: E402
from navros.pt.model import Navros, weighted_xent  # noqa: E402

CFG = NavrosConfig(vocab=13, d=16, n_heads=2, ffn=24, n_core=1, abacus=11)


def batch(rng, B=3, T=6):
    return dict(tokens=rng.integers(0, 13, (B, T)), targets=rng.integers(0, 13, (B, T)),
                weights=np.ones((B, T)), abacus=rng.integers(0, 11, (B, T)))


def worst(model, b, **kw):
    _, rep = directional_check(model, b, eps_sweep=(1e-3, 1e-4, 1e-5), **kw)
    return max(v["best_err"] for v in rep.values())


def test_gradcheck_pasa():
    m = NavrosNP(CFG, seed=0)
    assert worst(m, batch(np.random.default_rng(0)), r=4) < 1e-6


def test_gradcheck_detecta_bug_en_rmsnorm(monkeypatch):
    def buggy(dy, cache):  # olvida el término de la media
        xhat, inv, g = cache
        dg = None if g is None else np.sum(dy * xhat, axis=tuple(range(dy.ndim - 1)))
        return inv * (dy * g if g is not None else dy), dg
    monkeypatch.setattr(L, "rmsnorm_bwd", buggy)
    import navros.oracle.model as M
    monkeypatch.setattr(M, "rmsnorm_bwd", buggy)
    m = NavrosNP(CFG, seed=0)
    assert worst(m, batch(np.random.default_rng(0)), r=4) > 1e-3


def test_gradcheck_detecta_bug_en_rope(monkeypatch):
    monkeypatch.setattr(L, "rope_bwd", lambda dy, cos, sin: dy)  # finge que RoPE es la identidad
    m = NavrosNP(CFG, seed=0)
    assert worst(m, batch(np.random.default_rng(0)), r=4) > 1e-3


def test_truncado_no_es_el_gradiente_completo():
    """El gradiente truncado debe diferir del de la pérdida completa (si no, la prueba no discrimina)."""
    m = NavrosNP(CFG, seed=0)
    b = batch(np.random.default_rng(0))
    _, Gt, _ = m.loss_and_grads(b, r=6, k=2)
    _, Gf, _ = m.loss_and_grads(b, r=6)
    assert np.max(np.abs(Gt["core.0.0.wq"] - Gf["core.0.0.wq"])) > 1e-4


@pytest.mark.parametrize("perturb", [0.0, 1e-3])
def test_port_detecta_discrepancia(perturb):
    rng = np.random.default_rng(0)
    npm = NavrosNP(CFG, seed=0)
    b = batch(rng)
    _, G, _ = npm.loss_and_grads(b, r=4)
    m = Navros(CFG).double()
    m.load_numpy(npm.P)
    with torch.no_grad():
        m.core[0][0].wv[0, 0] += perturb
    logits = m(torch.from_numpy(b["tokens"]), torch.from_numpy(b["abacus"]), r=4)
    weighted_xent(logits, torch.from_numpy(b["targets"]), torch.from_numpy(b["weights"])).backward()
    err = max(float(np.max(np.abs(G[n] - p.grad.numpy())) / np.max(np.abs(G[n]))) for n, p in m.named_parameters())
    assert (err < 1e-12) if perturb == 0 else (err > 1e-6)


# --- Tareas: el verificador independiente tiene que poder fallar -------------------

def test_suma_verificada():
    from navros.tasks import suma
    rng = np.random.default_rng(0)
    for n in (1, 5, 17):
        tok, tgt, _, _ = suma.generate(rng, 200, n, 64, True)
        suma.verify(tok, tgt, n)


def test_suma_verificador_detecta_error():
    from navros.tasks import suma
    tok, tgt, _, _ = suma.generate(np.random.default_rng(0), 50, 6, 64, True)
    tgt[3, 8] = (tgt[3, 8] + 1) % 10
    with pytest.raises(AssertionError):
        suma.verify(tok, tgt, 6)


def test_expr_verificada():
    from navros.tasks import expr
    rng = np.random.default_rng(0)
    for n in (1, 3, 8, 32):
        for _ in range(100):
            expr.make_example(rng, n)


def test_expr_verificador_detecta_asociatividad(monkeypatch):
    """Un impresor que omite el paréntesis de a-(b-c) da un valor total a veces correcto,
    pero las etiquetas por subexpresión no coinciden con el parse de Python: debe saltar."""
    from navros.tasks import expr
    monkeypatch.setattr(expr, "PREC", {"+": 1, "-": 1, "*": 2})
    orig = expr.render

    def buggy(node, chars, labels):
        if isinstance(node, int):
            return orig(node, chars, labels)
        op, l, r = node
        for child, side in ((l, "L"), (r, "R")):
            if side == "R":
                chars.append(op)
                labels.append(expr.value(node))
            paren = not isinstance(child, int) and expr.PREC[child[0]] < expr.PREC[op]  # bug: < en vez de <=
            if paren:
                chars.append("(")
                labels.append(-1)
            buggy(child, chars, labels)
            if paren:
                chars.append(")")
                labels.append(expr.value(child))
    monkeypatch.setattr(expr, "render", buggy)
    rng = np.random.default_rng(0)
    with pytest.raises(AssertionError):
        for _ in range(200):
            expr.make_example(rng, 6)


def test_expr_ejemplo():
    from navros.tasks import expr
    s = "3+4*(2-7)"
    lab = expr.python_labels(s)
    # '+' raíz: 3+4*(-5) = -17 → 3 ; '*' : -20 → 0 ; '-' : -5 → 5 ; ')' : 5 ; '=' : 3
    assert lab == {1: 3, 3: 0, 6: 5, 8: 5, 9: 3}
