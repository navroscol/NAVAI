"""Superposición de conceptos: ¿caben más rasgos en el mismo recurso si el estado es bidimensional?

Modelo de juguete (Elhage et al. 2022): m rasgos dispersos x ∈ [0,1]^m (cada uno activo con
probabilidad 1-S) se comprimen a un estado de n NÚMEROS REALES y se reconstruyen con ReLU.
Se comparan tres geometrías del estado, con el mismo presupuesto de n reales:

  real      h = W x,            W ∈ ℝ^{n×m}          (el juguete original)
  complejo  h = W x,            W ∈ ℂ^{(n/2)×m}      (bidimensional por fase: cada coordenada es un plano)
  matriz    H = Σ_i x_i u_i v_iᵀ, u_i ∈ ℝ^a, v_i ∈ ℝ^b, ab = n   (bidimensional por rejilla: cada
            concepto es un producto exterior y el estado es una matriz a×b)

Lectura: x̂_i = ReLU(s_i + b_i) con s_i = ⟨W_i, h⟩ (real), s_i = |⟨W_i, h⟩| (complejo: el módulo, para
que la fase cuente; con Re⟨·,·⟩ sería idéntico al real) y s_i = u_iᵀ H v_i (matriz: la interferencia
entre i y j es el PRODUCTO (u_i·u_j)(v_i·v_j), no un solo coseno).

VECTORIZACIÓN: todos los modelos del barrido (dispersión × m/n × semilla) se entrenan A LA VEZ
como un tensor por lotes (K, ·, ·): una sola secuencia de einsum por paso, sin bucles en Python
sobre modelos. En GPU el barrido entero es una llamada.

Métricas por modelo:
  perdida       pérdida ponderada por importancia (I_i = 0.9^i), como en el original
  representados fracción de rasgos con ||W_i|| > 0.5 (el modelo "decidió" guardarlos)
  dims_por_rasgo n / Σ_i ||W_i||² (Elhage): <1 significa superposición
  interferencia media de |⟨Ŵ_i, W_j⟩| sobre pares i≠j de rasgos representados
  error_k       error de reconstrucción con exactamente k rasgos activos a la vez, k=1..8, elegidos
                SOLO entre los IMPORTANTES (los 32 primeros, importancia ≥ 0,04): mide interferencia
  error_k_todos lo mismo eligiendo entre todos los m rasgos: mide cobertura (versión 1)

Nombres de geometría: "real", "complejo", "matriz" (rango 1) o "matriz-r4" (cada concepto es una suma
de 4 productos exteriores: 2·4·√n parámetros; con r = √n/2 iguala los n del vector). El sufijo "-lr3"
entrena con LR 1e-3 en vez de 1e-2 (p. ej. "real-lr3").
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

AQUI = Path(__file__).resolve().parent
GEOMETRIAS = ("real", "complejo", "matriz")


def barrido(n: int, ratios=(2, 4, 8), dispersiones=(0.0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99), semillas=(0, 1, 2)):
    return [dict(n=n, m=n * r, S=S, semilla=s) for r in ratios for S in dispersiones for s in semillas]


class Lote(torch.nn.Module):
    """K modelos de juguete entrenados a la vez. Todos comparten n y m; S varía por modelo."""

    def __init__(self, geometria: str, K: int, n: int, m: int, dispositivo):
        super().__init__()
        self.nombre, self.K, self.n, self.m = geometria, K, n, m
        partes = geometria.split("-")
        geometria = partes[0]
        self.geometria = geometria
        self.r = next((int(x[1:]) for x in partes[1:] if x.startswith("r")), 1)
        g = torch.Generator(device="cpu").manual_seed(0)
        if geometria == "real":
            self.W = torch.nn.Parameter(torch.randn(K, n, m, generator=g) / math.sqrt(n))
        elif geometria == "complejo":
            assert n % 2 == 0
            self.W = torch.nn.Parameter(torch.randn(K, n // 2, m, 2, generator=g) / math.sqrt(n))  # (re, im)
        elif geometria == "matriz":
            a = int(round(math.sqrt(n)))
            assert a * a == n, "n debe ser cuadrado perfecto para la geometría matriz"
            self.a = a
            esc = 1 / math.sqrt(a) / self.r ** 0.25
            self.U = torch.nn.Parameter(torch.randn(K, a, self.r, m, generator=g) * esc)
            self.V = torch.nn.Parameter(torch.randn(K, a, self.r, m, generator=g) * esc)
        else:
            raise ValueError(geometria)
        self.b = torch.nn.Parameter(torch.zeros(K, m))
        self.to(dispositivo)

    def direcciones(self) -> torch.Tensor:
        """Vector real de n componentes por rasgo: (K, n, m). Sirve para normas e interferencias."""
        if self.geometria == "real":
            return self.W
        if self.geometria == "complejo":
            return torch.cat([self.W[..., 0], self.W[..., 1]], dim=1)
        return torch.einsum("karm,kbrm->kabm", self.U, self.V).reshape(self.K, self.n, self.m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (K, B, m) → x̂: (K, B, m)."""
        if self.geometria == "matriz":
            # H = Σ_i x_i u_i v_iᵀ  (K, B, a, a);  lectura ⟨u_i v_iᵀ, H⟩ = u_iᵀ H v_i
            H = torch.einsum("kbm,karm,kcrm->kbac", x, self.U, self.V)
            s = torch.einsum("karm,kbac,kcrm->kbm", self.U, H, self.V)
        elif self.geometria == "complejo":
            # h = W x ∈ ℂ^{n/2};  lectura por MÓDULO: |⟨W_i, h⟩| = |Σ_j x_j ⟨W_i, W_j⟩|. Las interferencias
            # de rasgos con fases distintas se suman de forma incoherente (pueden cancelarse). Con lectura
            # lineal Re⟨·,·⟩ esta geometría sería idéntica a la real con n dimensiones.
            Wc = torch.complex(self.W[..., 0], self.W[..., 1])
            h = torch.einsum("knm,kbm->kbn", Wc, x.to(Wc.dtype))
            z = torch.einsum("knm,kbn->kbm", Wc.conj(), h)
            s = torch.sqrt(z.real ** 2 + z.imag ** 2 + 1e-12)
        else:
            D = self.direcciones()
            h = torch.einsum("knm,kbm->kbn", D, x)
            s = torch.einsum("knm,kbn->kbm", D, h)
        return torch.relu(s + self.b[:, None, :])


def muestras(K: int, B: int, m: int, S: torch.Tensor, gen: torch.Generator, dispositivo) -> torch.Tensor:
    activo = torch.rand(K, B, m, generator=gen, device=dispositivo) > S[:, None, None]
    return activo * torch.rand(K, B, m, generator=gen, device=dispositivo)


def muestras_k(K: int, B: int, m: int, k: int, gen: torch.Generator, dispositivo, entre: int | None = None) -> torch.Tensor:
    """Exactamente k rasgos activos por muestra, elegidos al azar entre los `entre` primeros (todos si None)."""
    entre = m if entre is None else min(entre, m)
    orden = torch.rand(K, B, entre, generator=gen, device=dispositivo).argsort(dim=-1)
    activo = torch.zeros(K, B, m, dtype=torch.bool, device=dispositivo)
    activo[:, :, :entre] = orden < k
    return activo * torch.rand(K, B, m, generator=gen, device=dispositivo)


def entrenar(geometria: str, cfgs: list[dict], pasos: int, lote: int, dispositivo, log=print) -> list[dict]:
    n, m = cfgs[0]["n"], cfgs[0]["m"]
    assert all(c["n"] == n and c["m"] == m for c in cfgs)
    K = len(cfgs)
    S = torch.tensor([c["S"] for c in cfgs], device=dispositivo)
    I = (0.9 ** torch.arange(m, device=dispositivo))[None, None, :]  # importancia por rasgo
    modelo = Lote(geometria, K, n, m, dispositivo)
    with torch.no_grad():  # semilla propia por modelo: cada bloque k se inicializa con su generador
        for k, c in enumerate(cfgs):
            g = torch.Generator(device="cpu").manual_seed(1000 * (c["semilla"] + 1) + k)
            for nombre, p in modelo.named_parameters():
                if nombre != "b":
                    escala = 1 / math.sqrt(n) if nombre == "W" else 1 / math.sqrt(modelo.a) / modelo.r ** 0.25
                    p[k] = (torch.randn(p[k].shape, generator=g) * escala).to(dispositivo)
    lr = 1e-3 if "lr3" in geometria.split("-") else 1e-2
    opt = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / pasos)))
    gen = torch.Generator(device=dispositivo).manual_seed(7)
    t0 = time.time()
    for paso in range(pasos):
        x = muestras(K, lote, m, S, gen, dispositivo)
        perdida_k = (I * (modelo(x) - x) ** 2).mean(dim=(1, 2))  # (K,)
        opt.zero_grad(set_to_none=True)
        perdida_k.sum().backward()
        opt.step()
        sched.step()
        if paso % 1000 == 0 or paso == pasos - 1:
            log(f"  {geometria} n={n} m={m} paso {paso} pérdida media {perdida_k.mean().item():.5f} ({time.time() - t0:.0f}s)")
    # métricas
    with torch.no_grad():
        D = modelo.direcciones()                                    # (K, n, m)
        normas = D.norm(dim=1)                                      # (K, m)
        rep = normas > 0.5
        Dn = D / (normas[:, None, :] + 1e-8)
        G = torch.einsum("knm,knj->kmj", Dn, D).abs()               # |⟨Ŵ_i, W_j⟩|
        ojo = torch.eye(m, device=dispositivo, dtype=torch.bool)[None]
        par = rep[:, :, None] & rep[:, None, :] & ~ojo
        interf = (G * par).sum(dim=(1, 2)) / par.sum(dim=(1, 2)).clamp(min=1)
        dims = n / (normas ** 2).sum(dim=1).clamp(min=1e-8)
        x = muestras(K, 4096, m, S, gen, dispositivo)
        perdida = (I * (modelo(x) - x) ** 2).mean(dim=(1, 2))
        error_k, error_k_todos = {}, {}
        for k in range(1, 9):
            xk = muestras_k(K, 2048, m, k, gen, dispositivo, entre=32)   # solo rasgos importantes
            error_k[k] = ((modelo(xk) - xk) ** 2).sum(-1).mean(-1)      # error cuadrático total por muestra
            xk = muestras_k(K, 2048, m, k, gen, dispositivo)             # entre todos (versión 1)
            error_k_todos[k] = ((modelo(xk) - xk) ** 2).sum(-1).mean(-1)
    out = []
    for i, c in enumerate(cfgs):
        out.append(dict(c, geometria=geometria, pasos=pasos, perdida=perdida[i].item(),
                        representados=rep[i].float().mean().item(), dims_por_rasgo=dims[i].item(),
                        interferencia=interf[i].item(), error_k={k: v[i].item() for k, v in error_k.items()},
                        error_k_todos={k: v[i].item() for k, v in error_k_todos.items()},
                        parametros_por_rasgo=sum(p.numel() for nb, p in modelo.named_parameters() if nb != "b") // (K * m),
                        lr=lr, segundos=round(time.time() - t0, 1)))
    return out


def correr_todo(n: int, pasos: int, lote: int, ratios, dispersiones, semillas, geometrias, log=print) -> list[dict]:
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"dispositivo {dispositivo}, n={n}")
    res = []
    for geometria in geometrias:
        for r in ratios:
            cfgs = [c for c in barrido(n, (r,), dispersiones, semillas)]
            res += entrenar(geometria, cfgs, pasos, lote, dispositivo, log)
    return res


# ----------------------------------------------------------------------------- tablas
def tablas(res: list[dict]) -> str:
    import numpy as np
    from collections import defaultdict
    out = []
    n = res[0]["n"]
    geoms = list(dict.fromkeys(r["geometria"] for r in res))
    for metrica, titulo in [("representados", "fracción de rasgos representados"), ("dims_por_rasgo", "dimensiones por rasgo (n / Σ‖W_i‖²)"),
                            ("interferencia", "interferencia media |⟨Ŵ_i, W_j⟩| entre rasgos representados"), ("perdida", "pérdida ponderada")]:
        out.append(f"### {titulo} — n={n} reales, media ± desv. típica sobre semillas\n")
        Ss = sorted({r['S'] for r in res})
        out.append("| geometría | m/n | " + " | ".join(f"S={S}" for S in Ss) + " |")
        out.append("|---|---|" + "---|" * len(Ss))
        grupos = defaultdict(list)
        for r in res:
            grupos[(r["geometria"], r["m"] // n, r["S"])].append(r[metrica])
        for g in geoms:
            for ratio in sorted({r["m"] // n for r in res}):
                if (g, ratio, Ss[0]) not in grupos:
                    continue
                celdas = [f"{np.mean(grupos[(g, ratio, S)]):.3f} ± {np.std(grupos[(g, ratio, S)]):.3f}" for S in Ss]
                out.append(f"| {g} | {ratio} | " + " | ".join(celdas) + " |")
        out.append("")
    for clave, titulo in [("error_k", "entre los 32 rasgos IMPORTANTES (mide interferencia)"), ("error_k_todos", "entre todos los rasgos (mide cobertura)")]:
        if clave not in res[0]:
            continue
        out.append(f"### error cuadrático por muestra con exactamente k conceptos activos a la vez, {titulo} — m/n=4, S=0.9, n={n}\n")
        out.append("| geometría | parámetros/rasgo | " + " | ".join(f"k={k}" for k in range(1, 9)) + " |")
        out.append("|---|---|" + "---|" * 8)
        for g in geoms:
            rs = [r for r in res if r["geometria"] == g and r["m"] // n == 4 and abs(r["S"] - 0.9) < 1e-9]
            if rs:
                v = lambda r, k: r[clave][str(k)] if str(k) in r[clave] else r[clave][k]
                out.append(f"| {g} | {rs[0].get('parametros_por_rasgo', '')} | " + " | ".join(f"{np.mean([v(r, k) for r in rs]):.3f}" for k in range(1, 9)) + " |")
        out.append("")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--pasos", type=int, default=10_000)
    ap.add_argument("--lote", type=int, default=1024)
    ap.add_argument("--rapido", action="store_true", help="barrido mínimo para comprobar que corre")
    ap.add_argument("--geometrias", default="real,complejo,matriz", help="lista separada por comas; p. ej. real,real-lr3,complejo,matriz,matriz-r2,matriz-r4")
    ap.add_argument("--salida", default=str(AQUI / "resultados" / "conceptos.json"))
    a = ap.parse_args()
    if a.rapido:
        res = correr_todo(16, 300, 256, (2, 4), (0.0, 0.9), (0,), a.geometrias.split(","))
    else:
        res = correr_todo(a.n, a.pasos, a.lote, (2, 4, 8), (0.0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99), (0, 1, 2), a.geometrias.split(","))
    Path(a.salida).parent.mkdir(parents=True, exist_ok=True)
    Path(a.salida).write_text(json.dumps(res, indent=1))
    print(tablas(res))
