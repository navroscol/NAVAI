"""Modelos del banco de pruebas de superposición.

La idea central, en su forma clásica y computable: el estado de la secuencia no es un vector de
activaciones sino una AMPLITUD sobre χ estados internos a la vez (superposición). Cada símbolo x
aplica una matriz A[x] ∈ C^{χ×χ}; la secuencia completa es un producto de matrices (MPS / red de
tensores) y la lectura sigue la regla de Born |amplitud|², de modo que las hipótesis internas
interfieren constructiva o destructivamente antes de emitir el bit.

Variantes (todas comparten el mismo esqueleto de contracción izquierda/derecha):
  mps-real      matrices reales, lectura lineal (sin interferencia).
  mps-complejo  matrices complejas libres, lectura de Born.
  mps-unitario  A[x] = Cayley(H[x]) unitarias: función de onda simulada de norma constante.
  mps-fourier   A[x] = V diag(e^{iθ[x]}) V†, base propia COMPARTIDA: todas las A conmutan.
                Es la "serie de Fourier" pura: el estado es Σ_k c_k e^{i Σ_t θ_k[x_t]}.
  Compuerta de espigas (LIF con gradiente sustituto) entre dos capas MPS.
  Dos cabezas con crítica cruzada: cada cabeza corrige su amplitud tras ver la de la otra,
                y la respuesta final es la interferencia de ambas.
Referencia: Transformer codificador (bidireccional) con RoPE o con ábaco (posiciones desplazadas
al azar en entrenamiento, como en navros/).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB = 3  # 0, 1, PAD


# ----------------------------------------------------------------------------- utilidades complejas
def cayley(B: torch.Tensor) -> torch.Tensor:
    """B libre (compleja) → H = B + B† hermitiana → U = (I - iH)(I + iH)^{-1} unitaria."""
    H = B + B.conj().transpose(-1, -2)
    I = torch.eye(B.shape[-1], dtype=B.dtype, device=B.device)
    return torch.linalg.solve(I + 1j * H, I - 1j * H, left=False)


def _normaliza(v: torch.Tensor) -> torch.Tensor:
    return v / (v.norm(dim=-1, keepdim=True) + 1e-8)


# ----------------------------------------------------------------------------- capa MPS
class CapaMPS(nn.Module):
    """Contrae la secuencia por la izquierda y por la derecha y devuelve, por posición,
    las amplitudes (o valores reales) de cada clase: s_t[c] = L_t · M[c] · R_{t+1}.

    Si `entrada_continua`, la matriz de cada posición es A(f_t) = A_0 + Σ_j f_tj A_j
    (mezcla lineal de matrices) en vez de A[x_t]; sirve para apilar capas."""

    def __init__(self, chi: int, n_clases: int, variante: str, n_simbolos: int = VOCAB,
                 entrada_continua: bool = False, d_entrada: int = 0):
        super().__init__()
        self.chi, self.variante, self.n_clases = chi, variante, n_clases
        self.compleja = variante != "real"
        self.entrada_continua = entrada_continua
        dt = torch.cfloat if self.compleja else torch.float32
        n_mat = (1 + d_entrada) if entrada_continua else n_simbolos
        esc = 1.0 / math.sqrt(chi)
        self.A = nn.Parameter(torch.randn(n_mat, chi, chi, dtype=dt) * esc)
        if variante == "fourier":
            self.V = nn.Parameter(torch.randn(chi, chi, dtype=torch.cfloat) * esc)  # base propia compartida
            self.theta = nn.Parameter(torch.rand(n_mat, chi) * 2 * math.pi)        # fases por símbolo
        self.izq = nn.Parameter(torch.randn(chi, dtype=dt) * esc)
        self.der = nn.Parameter(torch.randn(chi, dtype=dt) * esc)
        self.M = nn.Parameter(torch.randn(n_clases, chi, chi, dtype=dt) * esc)

    def matrices(self) -> torch.Tensor:
        if self.variante in ("real", "complejo"):
            return self.A
        if self.variante == "unitario":
            return cayley(self.A)
        if self.variante == "fourier":
            V = cayley(self.V)
            D = torch.exp(1j * self.theta)  # (n_mat, chi)
            return V.unsqueeze(0) * D.unsqueeze(1) @ V.conj().transpose(-1, -2).unsqueeze(0)
        raise ValueError(self.variante)

    def por_posicion(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T) símbolos o (B, T, d) continuo → matrices (B, T, chi, chi)."""
        A = self.matrices()
        if not self.entrada_continua:
            return A[x]
        f = x.to(A.dtype) if self.compleja else x
        return A[0] + torch.einsum("ntj,jab->ntab", f, A[1:])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Devuelve s: (B, T, n_clases) amplitudes complejas (o reales en la variante real)."""
        Ax = self.por_posicion(x)  # (B, T, χ, χ)
        B, T = Ax.shape[:2]
        L = [_normaliza(self.izq).expand(B, -1)]
        for t in range(T):
            L.append(_normaliza(torch.einsum("ba,bac->bc", L[-1], Ax[:, t])))
        R = [_normaliza(self.der).expand(B, -1)]
        for t in reversed(range(T)):
            R.append(_normaliza(torch.einsum("bac,bc->ba", Ax[:, t], R[-1])))
        R = R[::-1]  # R[t] = entorno derecho que empieza en la posición t
        # La lectura de la posición t vive en el enlace (t, t+1): entorno izquierdo que YA incluye x_t
        # y entorno derecho a partir de t+1. Así cada bit de salida ve la secuencia entera.
        Lt = torch.stack(L[1:], dim=1)   # (B, T, χ): entorno izquierdo hasta t inclusive
        Rt = torch.stack(R[1:], dim=1)   # (B, T, χ): entorno derecho después de t
        return torch.einsum("nta,cab,ntb->ntc", Lt, self.M, Rt)

    def estado_final(self, x: torch.Tensor) -> torch.Tensor:
        Ax = self.por_posicion(x)
        v = _normaliza(self.izq).expand(Ax.shape[0], -1)
        for t in range(Ax.shape[1]):
            v = _normaliza(torch.einsum("ba,bac->bc", v, Ax[:, t]))
        return v


def born(s: torch.Tensor) -> torch.Tensor:
    """Amplitudes → log-probabilidades por la regla de Born (interferencia)."""
    if s.is_complex():
        p = s.real ** 2 + s.imag ** 2
        return torch.log(p + 1e-9) - torch.log(p.sum(-1, keepdim=True) + 1e-9)
    return F.log_softmax(s, dim=-1)


class ModeloMPS(nn.Module):
    """Una capa MPS con lectura de Born (o lineal si es real). Para 'parada' clasifica el estado final."""

    def __init__(self, chi: int, variante: str, secuencia: bool):
        super().__init__()
        self.secuencia = secuencia
        self.capa = CapaMPS(chi, 2, variante)
        if not secuencia:
            dt = torch.cfloat if variante != "real" else torch.float32
            self.lectura = nn.Parameter(torch.randn(2, chi, dtype=dt) / math.sqrt(chi))

    def forward(self, x):
        if self.secuencia:
            return born(self.capa(x))
        v = self.capa.estado_final(x)
        return born(torch.einsum("ba,ca->bc", v, self.lectura))


# ----------------------------------------------------------------------------- espigas (LIF)
class _Escalon(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v):
        ctx.save_for_backward(v)
        return (v > 0).to(v.dtype)

    @staticmethod
    def backward(ctx, g):
        (v,) = ctx.saved_tensors
        sig = torch.sigmoid(4.0 * v)
        return g * 4.0 * sig * (1 - sig)  # gradiente sustituto


class CompuertaEspigas(nn.Module):
    """Integra-y-dispara con fuga a lo largo de la secuencia. Solo los canales cuya energía
    acumulada cruza el umbral 'disparan' y pasan a la capa siguiente; el resto queda en silencio.
    Devuelve la señal filtrada y la tasa de disparo (fracción de cómputo activo)."""

    def __init__(self, d: int, beta: float = 0.7):
        super().__init__()
        self.beta = beta
        self.umbral = nn.Parameter(torch.ones(d) * 0.5)

    def forward(self, f: torch.Tensor):
        B, T, d = f.shape
        e = f ** 2
        v = torch.zeros(B, d, dtype=f.dtype, device=f.device)
        salidas, tasa = [], 0.0
        for t in range(T):
            v = self.beta * v + e[:, t]
            s = _Escalon.apply(v - F.softplus(self.umbral))
            v = v * (1 - s)  # reinicio tras disparar
            salidas.append(s * f[:, t])
            tasa = tasa + s.mean()
        return torch.stack(salidas, dim=1), tasa / T


class ModeloMPSEspigas(nn.Module):
    """Dos capas MPS con una compuerta de espigas entre ambas. Con `compuerta=False` es el control:
    las mismas dos capas sin compuerta (la señal pasa entera)."""

    def __init__(self, chi: int, variante: str, d_oculto: int = 8, compuerta: bool = True):
        super().__init__()
        self.capa1 = CapaMPS(chi, d_oculto, variante)
        self.espigas = CompuertaEspigas(2 * d_oculto) if compuerta else None
        self.capa2 = CapaMPS(chi, 2, variante, entrada_continua=True, d_entrada=2 * d_oculto)
        self.tasa = torch.tensor(1.0)

    def forward(self, x):
        s = self.capa1(x)
        f = torch.cat([s.real, s.imag], dim=-1) if s.is_complex() else torch.cat([s, s * 0], -1)
        if self.espigas is not None:
            f, self.tasa = self.espigas(f)
        return born(self.capa2(f))


# ----------------------------------------------------------------------------- dos cabezas, crítica cruzada
class ModeloDual(nn.Module):
    """Dos cabezas MPS (A y B). Con crítica: cada cabeza ve la propuesta de la otra (sus amplitudes,
    módulo y fase) y emite una corrección compleja a la suya; la respuesta final es la suma
    (interferencia) de las dos amplitudes corregidas. Sin crítica: suma directa."""

    def __init__(self, chi: int, variante: str, critica: bool):
        super().__init__()
        self.critica = critica
        self.A = CapaMPS(chi, 2, variante)
        self.B = CapaMPS(chi, 2, variante)
        if critica:
            self.crit_A = nn.Sequential(nn.Linear(6, 16), nn.GELU(), nn.Linear(16, 4))
            self.crit_B = nn.Sequential(nn.Linear(6, 16), nn.GELU(), nn.Linear(16, 4))

    @staticmethod
    def _rasgos(s):
        return torch.cat([s.real, s.imag, s.real ** 2 + s.imag ** 2], dim=-1)

    @staticmethod
    def _corrige(s, c):
        return s + torch.complex(c[..., :2], c[..., 2:])

    def forward(self, x):
        sA, sB = self.A(x), self.B(x)
        self.parciales = (born(sA), born(sB))
        if self.critica:
            sA2 = self._corrige(sA, self.crit_A(self._rasgos(sB)))  # A critica a B y corrige su propuesta
            sB2 = self._corrige(sB, self.crit_B(self._rasgos(sA)))
            return born(sA2 + sB2)
        return born(sA + sB)


# ----------------------------------------------------------------------------- Transformer de referencia
def _rope(q, k, T, device):
    d = q.shape[-1]
    pos = torch.arange(T, device=device).float()
    inv = 1.0 / (10000 ** (torch.arange(0, d, 2, device=device).float() / d))
    ang = pos[:, None] * inv[None]
    cos, sin = ang.cos()[None, None], ang.sin()[None, None]

    def rot(x):
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)
    return rot(q), rot(k)


class _Bloque(nn.Module):
    def __init__(self, d, h, rope):
        super().__init__()
        self.h, self.rope = h, rope
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        B, T, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        if self.rope:
            q, k = _rope(q, k, T, x.device)
        a = F.scaled_dot_product_attention(q, k, v)
        x = x + self.o(a.transpose(1, 2).reshape(B, T, d))
        return x + self.ffn(self.n2(x))


class Transformer(nn.Module):
    def __init__(self, d: int, capas: int, cabezas: int, posicion: str, secuencia: bool, max_pos: int = 128):
        super().__init__()
        self.posicion, self.secuencia, self.max_pos = posicion, secuencia, max_pos
        self.emb = nn.Embedding(VOCAB, d)
        if posicion == "abaco":
            self.pos = nn.Embedding(max_pos, d)
        self.bloques = nn.ModuleList([_Bloque(d, cabezas, rope=(posicion == "rope")) for _ in range(capas)])
        self.norma = nn.LayerNorm(d)
        self.salida = nn.Linear(d, 2)

    def forward(self, x):
        B, T = x.shape
        h = self.emb(x)
        if self.posicion == "abaco":
            desde = torch.randint(0, self.max_pos - T, (1,)).item() if self.training else 0
            h = h + self.pos(torch.arange(desde, desde + T, device=x.device))[None]
        for b in self.bloques:
            h = b(h)
        h = self.norma(h)
        if not self.secuencia:
            h = h.mean(1)
        return F.log_softmax(self.salida(h), dim=-1)


# ----------------------------------------------------------------------------- fábrica
def construir(nombre: str, chi: int, secuencia: bool) -> nn.Module:
    if nombre.startswith("mps-espigas"):
        return ModeloMPSEspigas(chi, "complejo")
    if nombre == "mps-2capas":  # control: dos capas sin compuerta
        return ModeloMPSEspigas(chi, "complejo", compuerta=False)
    if nombre == "mps-dual-critica":
        return ModeloDual(chi, "complejo", critica=True)
    if nombre == "mps-dual-suma":
        return ModeloDual(chi, "complejo", critica=False)
    if nombre.startswith("mps-"):
        return ModeloMPS(chi, nombre.split("-")[1], secuencia)
    if nombre.startswith("tf-"):  # χ=16 → d=64 y 2 capas; χ=32 → d=128 y 4 capas
        return Transformer(4 * chi, 2 if chi <= 16 else 4, 4, nombre.split("-")[1], secuencia)
    raise ValueError(nombre)


def n_parametros(m: nn.Module) -> int:
    return sum(p.numel() * (2 if p.is_complex() else 1) for p in m.parameters())


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randint(0, 3, (4, 10))
    for nombre in ["mps-real", "mps-complejo", "mps-unitario", "mps-fourier", "mps-espigas", "mps-dual-critica",
                   "mps-dual-suma", "tf-rope", "tf-abaco"]:
        m = construir(nombre, 16, True)
        y = m(x)
        assert y.shape == (4, 10, 2) and torch.isfinite(y).all(), nombre
        assert torch.allclose(y.exp().sum(-1), torch.ones(4, 10), atol=1e-4), nombre
        y.sum().backward()
        print(f"{nombre:18s} params={n_parametros(m):7d} salida={tuple(y.shape)}")
    U = cayley(torch.randn(8, 8, dtype=torch.cfloat))
    assert torch.allclose(U @ U.conj().T, torch.eye(8, dtype=torch.cfloat), atol=1e-5)
    m = construir("mps-complejo", 16, False)
    print("parada:", m(x).shape)
    print("ok")
