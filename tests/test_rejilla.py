"""Rejilla (atención lineal con estado matricial): la forma cuadrática enmascarada es la recurrencia
S_t = S_{t-1} + φ(k_t)v_tᵀ, y la caché de generación coincide con el forward completo."""
import torch

from navros.config import NavrosConfig
from navros.generate import step
from navros.pt.model import Navros, atencion_rejilla, rasgos_rejilla


def test_forma_cuadratica_igual_a_recurrencia():
    torch.manual_seed(0)
    B, H, T, hd = 2, 3, 9, 8
    q, k, v = torch.randn(B, H, T, hd), torch.randn(B, H, T, hd), torch.randn(B, H, T, hd)
    o, S, z = atencion_rejilla(q, k, v)
    qf, kf = rasgos_rejilla(q), rasgos_rejilla(k)
    S_t, z_t, outs = torch.zeros(B, H, hd, hd), torch.zeros(B, H, hd), []
    for t in range(T):
        S_t = S_t + kf[:, :, t, :, None] * v[:, :, t, None, :]
        z_t = z_t + kf[:, :, t]
        num = (qf[:, :, t, :, None] * S_t).sum(2)
        den = (qf[:, :, t] * z_t).sum(-1, keepdim=True)
        outs.append(num / (den + 1e-6))
    ref = torch.stack(outs, 2)
    assert torch.allclose(o, ref, atol=1e-5), (o - ref).abs().max()
    assert torch.allclose(S, S_t, atol=1e-5) and torch.allclose(z, z_t, atol=1e-5)


def test_estado_previo_equivale_a_secuencia_entera():
    torch.manual_seed(1)
    B, H, T, hd = 1, 2, 12, 6
    q, k, v = torch.randn(B, H, T, hd), torch.randn(B, H, T, hd), torch.randn(B, H, T, hd)
    o_todo, _, _ = atencion_rejilla(q, k, v)
    o1, S, z = atencion_rejilla(q[:, :, :5], k[:, :, :5], v[:, :, :5])
    o2, _, _ = atencion_rejilla(q[:, :, 5:], k[:, :, 5:], v[:, :, 5:], S=S, z=z)
    assert torch.allclose(torch.cat([o1, o2], 2), o_todo, atol=1e-5)


def test_cache_de_generacion_con_rejilla():
    torch.manual_seed(2)
    cfg = NavrosConfig(vocab=50, d=32, n_heads=4, n_pre=5, n_core=0, causal=True, grid_every=2, rope_frac=0.5)
    model = Navros(cfg).eval()
    assert [l.grid for l in model.pre] == [False, True, False, True, False]
    tokens = torch.randint(0, 50, (2, 10))
    with torch.no_grad():
        completo = model(tokens)
        caches = [{} for _ in model.pre]
        a = step(model, tokens[:, :6], caches, 0)
        b = step(model, tokens[:, 6:9], caches, 6)
        c = step(model, tokens[:, 9:], caches, 9)
    inc = torch.cat([a, b, c], 1)
    assert torch.allclose(inc, completo, atol=1e-4), (inc - completo).abs().max()


def test_causalidad():
    torch.manual_seed(3)
    cfg = NavrosConfig(vocab=50, d=16, n_heads=2, n_pre=3, n_core=0, causal=True, grid_every=1)  # todas rejilla salvo la 0
    model = Navros(cfg).eval()
    x = torch.randint(0, 50, (1, 8))
    y = x.clone(); y[0, 6:] = (y[0, 6:] + 7) % 50
    with torch.no_grad():
        assert torch.allclose(model(x)[:, :6], model(y)[:, :6], atol=1e-5)
