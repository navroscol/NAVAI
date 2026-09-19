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


# --- Caminos para TPU: deben dar lo mismo que los verificados ---------------------

def _train_steps(model, opt, rng, steps=5, r=4, k=2):
    for step in range(steps):
        b = batch(rng)
        opt.zero_grad(set_to_none=True)
        logits = model(torch.from_numpy(b["tokens"]), torch.from_numpy(b["abacus"]), r=r, k=k)
        weighted_xent(logits, torch.from_numpy(b["targets"]), torch.from_numpy(b["weights"])).backward()
        opt.step(1.0 - step / steps)
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def test_muon_escalares_tensor_igual_que_float():
    from navros.pt.optim import Muon
    out = []
    for ts in (False, True):
        torch.manual_seed(0)
        m = Navros(CFG).double()
        opt = Muon(m.named_parameters(), wd_muon=0.01, wd_adam=0.01, tensor_scalars=ts)
        out.append(_train_steps(m, opt, np.random.default_rng(0)))
    err = max(float((out[0][n] - out[1][n]).abs().max() / out[0][n].abs().max()) for n in out[0])
    assert err < 1e-6, err  # float32 en los escalares del camino tensor


def test_checkpointing_mismos_gradientes():
    import torch.utils.checkpoint as ckpt
    rng = np.random.default_rng(0)
    b = batch(rng)
    grads = []
    for use in (False, True):
        torch.manual_seed(0)
        m = Navros(CFG).double()
        if use:
            m.ckpt_fn = lambda f, *a: ckpt.checkpoint(f, *a, use_reentrant=False)
        logits = m(torch.from_numpy(b["tokens"]), torch.from_numpy(b["abacus"]), r=5, k=3)
        weighted_xent(logits, torch.from_numpy(b["targets"]), torch.from_numpy(b["weights"])).backward()
        grads.append({n: p.grad.clone() for n, p in m.named_parameters()})
    err = max(float((grads[0][n] - grads[1][n]).abs().max()) for n in grads[0])
    assert err < 1e-12, err


def test_hook_de_estado_se_llama():
    from navros.pt.optim import Muon
    seen = []
    m = Navros(CFG).double()
    opt = Muon(m.named_parameters(), state_hook=lambda z, p: seen.append(z.shape == p.shape))
    _train_steps(m, opt, np.random.default_rng(0), steps=1)
    n_muon = sum(1 for n, p in m.named_parameters() if p.ndim == 2 and n not in ("emb", "abaco"))
    n_adam = sum(1 for n, p in m.named_parameters()) - n_muon
    assert len(seen) == n_muon + 2 * n_adam and all(seen)


def test_escalador_salta_y_reduce_con_inf():
    from navros.lm_ddp import LossScaler
    p = torch.nn.Parameter(torch.ones(3))
    p.grad = torch.tensor([1.0, float("inf"), 0.0])
    sc = LossScaler(init=1024.0)
    ok, _ = sc.unscale_clip([p], 1.0)
    assert not ok and sc.scale == 512.0 and sc.skipped == 1
    p.grad = torch.tensor([512.0 * 3, 512.0 * 4, 0.0])  # gradiente real (3,4,0) escalado ×512 → norma 5
    ok, norm = sc.unscale_clip([p], 1.0)
    assert ok and abs(norm - 5.0) < 1e-5 and torch.allclose(p.grad, torch.tensor([0.6, 0.8, 0.0]), atol=1e-6)


def test_reparto_por_duenos_equilibrado_y_determinista():
    from navros.lm_ddp import partition
    m = Navros(NavrosConfig(vocab=50, d=32, n_heads=2, n_pre=2, n_core=0, causal=True))
    named = list(m.named_parameters())
    o1, loads = partition(named, 2)
    o2, _ = partition(list(reversed(named)), 2)
    assert o1 == o2 and set(o1.values()) == {0, 1}
    assert max(loads) / sum(loads) < 0.6
