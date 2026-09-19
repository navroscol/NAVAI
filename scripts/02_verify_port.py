"""Paso 4 del protocolo: el port PyTorch se verifica contra el oráculo NumPy.

Tres pruebas por configuración:
  A. Mismos pesos, misma entrada → pérdida y TODOS los gradientes (float64, atención manual y SDPA).
  B. Lo mismo con PyTorch en float32 (la precisión en la que se entrena en GPU).
  C. Trayectoria: N pasos de entrenamiento con Muon y con AdamW en ambos (float64),
     incluido el muestreo de r y el recorte de gradiente; se comparan los pesos finales.

Métrica: para cada tensor, max|a − b| / max|a| (error relativo a la escala del tensor).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navros.config import NavrosConfig  # noqa: E402
from navros.oracle.model import NavrosNP, sample_r  # noqa: E402
from navros.oracle.optim import MuonNP, clip_grads  # noqa: E402
from navros.pt.model import Navros, weighted_xent  # noqa: E402
from navros.pt.optim import Muon, clip_grad_norm  # noqa: E402

TOL64, TOL32 = 2e-5, 2e-5

CASES = [
    ("lm_fijo_causal", NavrosConfig(vocab=31, d=32, n_heads=4, n_pre=3, n_core=0, causal=True), 4, 12, None, None, False),
    ("razonador_bucle", NavrosConfig(vocab=13, d=32, n_heads=4, n_core=1, abacus=40, r_mean=5, k_bptt=3), 4, 10, 6, 3, True),
    ("razonador_bucle_2capas", NavrosConfig(vocab=17, d=48, n_heads=4, n_core=2, abacus=30, r_mean=4, k_bptt=4), 3, 9, 5, None, True),
    ("referencia_fija", NavrosConfig(vocab=13, d=32, n_heads=2, n_core=1, n_blocks=6, abacus=40), 4, 10, None, None, True),
    ("lm_recurrente", NavrosConfig(vocab=29, d=32, n_heads=4, n_pre=1, n_core=2, n_coda=1, causal=True, r_mean=3, k_bptt=2), 3, 11, 4, 2, False),
    ("sin_rope", NavrosConfig(vocab=13, d=24, n_heads=3, n_core=1, rope=False, abacus=20, r_mean=3, k_bptt=2), 4, 8, 3, 2, True),
]


def make_batch(cfg, B, T, rng, pad):
    b = dict(tokens=rng.integers(0, cfg.vocab, (B, T)), targets=rng.integers(0, cfg.vocab, (B, T)),
             weights=(rng.random((B, T)) > 0.3).astype(np.float64))
    b["weights"][0, 0] = 1.0
    if cfg.abacus:
        b["abacus"] = rng.integers(0, cfg.abacus, (B, T))
    if pad:
        valid = np.ones((B, T), dtype=bool)
        valid[0, -3:] = False
        valid[1, -1:] = False
        b["valid"] = valid
        b["weights"] = b["weights"] * valid
    return b


def to_torch(b, dtype, device="cpu"):
    out = dict(tokens=torch.from_numpy(b["tokens"]), targets=torch.from_numpy(b["targets"]),
               weights=torch.from_numpy(b["weights"]).to(dtype))
    out["abacus"] = torch.from_numpy(b["abacus"]) if "abacus" in b else None
    out["valid"] = torch.from_numpy(b["valid"]) if "valid" in b else None
    return {k: (v.to(device) if v is not None else None) for k, v in out.items()}


def pt_loss_grads(model, tb, r, k):
    model.zero_grad(set_to_none=True)
    logits = model(tb["tokens"], tb["abacus"], tb["valid"], r=r, k=k)
    loss = weighted_xent(logits, tb["targets"], tb["weights"])
    loss.backward()
    return float(loss.detach()), {n: p.grad.detach().double().cpu().numpy() for n, p in model.named_parameters()}


def rel_err(a: dict, b: dict):
    errs = {k: float(np.max(np.abs(a[k] - b[k])) / (np.max(np.abs(a[k])) + 1e-30)) for k in a}
    worst = max(errs, key=errs.get)
    return errs[worst], worst


def trajectory(cfg, B, T, pad, use_muon, steps, seed):
    """Entrena `steps` pasos en ambas implementaciones con los mismos datos y el mismo r."""
    npm = NavrosNP(cfg, seed=seed)
    ptm = Navros(cfg).double()
    ptm.load_numpy(npm.P)
    kw = dict(lr_muon=0.02, lr_adam=3e-3, wd_muon=0.01, wd_adam=0.01, use_muon=use_muon)
    opt_np = MuonNP(npm.P, **kw)
    opt_pt = Muon(ptm.named_parameters(), **kw)
    rng_data, rng_r = np.random.default_rng(seed + 1), np.random.default_rng(seed + 2)
    for step in range(steps):
        b = make_batch(cfg, B, T, rng_data, pad)
        r = sample_r(rng_r, cfg) if cfg.recurrent else None
        k = cfg.k_bptt if cfg.recurrent else None
        lr_mult = 1.0 - step / steps
        _, G, _ = npm.loss_and_grads(b, r=r, k=k)
        clip_grads(G, 1.0)
        opt_np.step(G, lr_mult)
        pt_loss_grads(ptm, to_torch(b, torch.float64), r, k)
        clip_grad_norm(ptm.parameters(), 1.0)
        opt_pt.step(lr_mult)
    P_pt = {n: p.detach().numpy() for n, p in ptm.named_parameters()}
    return rel_err(npm.P, P_pt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--out", default="results/verify_port.json")
    args = ap.parse_args()
    torch.manual_seed(0)
    rows, ok_all = [], True
    for i, (name, cfg, B, T, r, k, pad) in enumerate(CASES):
        rng = np.random.default_rng(i)
        npm = NavrosNP(cfg, seed=i)
        for v in npm.P.values():
            if v.ndim == 1:
                v += 0.1 * rng.standard_normal(v.shape)
        b = make_batch(cfg, B, T, rng, pad)
        l_np, G_np, _ = npm.loss_and_grads(b, r=r, k=k)

        res = dict(case=name, params=cfg.n_params())
        checks = [("f64_manual", torch.float64, True, "cpu"), ("f64_sdpa", torch.float64, False, "cpu"),
                  ("f32_sdpa", torch.float32, False, "cpu")]
        if torch.cuda.is_available():  # la GPU real donde se entrena
            checks.append(("cuda_f32_sdpa", torch.float32, False, "cuda"))
        for label, dtype, manual, dev in checks:
            m = Navros(cfg).to(dtype)
            m.load_numpy(npm.P)
            m = m.to(dev)
            m.manual_attn = manual
            l_pt, G_pt = pt_loss_grads(m, to_torch(b, dtype, dev), r, k)
            e, w = rel_err(G_np, G_pt)
            res[label] = dict(loss_diff=abs(l_np - l_pt) / abs(l_np), grad_err=e, worst_tensor=w)
        res["traj_muon"], res["traj_muon_worst"] = trajectory(cfg, B, T, pad, True, args.steps, 10 + i)
        res["traj_adamw"], res["traj_adamw_worst"] = trajectory(cfg, B, T, pad, False, args.steps, 20 + i)
        ok = (res["f64_manual"]["grad_err"] < TOL64 and res["f64_sdpa"]["grad_err"] < TOL64
              and res["f32_sdpa"]["grad_err"] < TOL32 and res["traj_muon"] < TOL64 and res["traj_adamw"] < TOL64
              and res.get("cuda_f32_sdpa", {"grad_err": 0.0})["grad_err"] < TOL32)
        res["ok"] = ok
        ok_all &= ok
        rows.append(res)
        print(f"{name:<24} f64 manual {res['f64_manual']['grad_err']:.1e} | f64 sdpa {res['f64_sdpa']['grad_err']:.1e} "
              f"| f32 sdpa {res['f32_sdpa']['grad_err']:.1e} ({res['f32_sdpa']['worst_tensor']}) "
              + (f"| cuda f32 {res['cuda_f32_sdpa']['grad_err']:.1e} " if "cuda_f32_sdpa" in res else "")
              + f"| trayectoria {args.steps} pasos: Muon {res['traj_muon']:.1e}, AdamW {res['traj_adamw']:.1e}  "
              f"{'OK' if ok else 'FALLO'}")
    print(f"\nRESUMEN: {'OK' if ok_all else 'FALLO'}  (tolerancia {TOL64:.0e})")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(dict(tol64=TOL64, tol32=TOL32, steps=args.steps, cases=rows), indent=1))
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
