"""Configuración compartida por el oráculo NumPy y el port PyTorch.

Un único modelo cubre los tres casos del proyecto:

  * LM estándar:        n_pre = L, n_core = 0                 (pila fija causal)
  * Razonador bucle:    n_pre = 0, n_core = c, n_blocks = 1    (mismos pesos, r iteraciones)
  * Referencia fija:    n_pre = 0, n_core = c, n_blocks = D    (D bloques distintos, r = D)
  * LM recurrente:      n_pre = p, n_core = c, n_blocks = 1, n_coda = q (preludio/núcleo/coda)

La referencia fija ejecuta exactamente el mismo grafo que el bucle (inyección de
entrada + norma sin parámetros tras cada bloque); la única diferencia es que cada
iteración tiene sus propios pesos y que r no se muestrea.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass
class NavrosConfig:
    vocab: int = 256
    d: int = 128                 # ancho C
    n_heads: int = 4
    ffn: int = 0                 # 0 → ~8/3·d redondeado a múltiplo de 32
    n_pre: int = 0               # capas de preludio (residual estándar)
    n_core: int = 1              # capas dentro de un bloque de núcleo
    n_blocks: int = 1            # 1 = recurrente (pesos compartidos); D = referencia fija
    n_coda: int = 0              # capas de coda
    causal: bool = False
    rope: bool = True
    rope_theta: float = 10000.0
    rope_frac: float = 1.0       # fracción de head_dim que rota (Phi-4-mini: 0,75); el resto pasa sin rotar
    rope_factors: tuple = ()     # LongRoPE: divisor por frecuencia (vacío = 1 en todas)
    rope_mscale: float = 1.0     # LongRoPE: cos y sin se multiplican por esto
    res_scale: float = 0.0       # 0 = 1/√(2L) en las ramas residuales; >0 = ese valor (modelos portados: 1)
    logit_scale_fixed: float = 0.0  # 0 = 1/√d; >0 = ese valor (modelos portados: 1)
    grid_every: int = 0          # rejilla: 0 = toda la atención es softmax; k>0 = softmax en las capas i % k == 0
                                 # y atención lineal con estado matricial (Σ φ(k)·vᵀ) en las demás
    grid_feature: str = "elu"    # mapa de rasgos de la rejilla: elu (elu(x)+1) | relu
    abacus: int = 0              # tamaño de la tabla ábaco (0 = sin ábaco)
    norm_eps: float = 1e-6
    # recurrencia (solo si n_blocks == 1 y n_core > 0)
    r_mean: float = 8.0          # media de r en entrenamiento
    r_sigma: float = 0.5         # sigma del log-normal
    r_max: int = 32              # tope de r muestreado
    k_bptt: int = 4              # BPTT truncado: gradiente por las últimas k iteraciones

    def __post_init__(self):
        if self.ffn == 0:
            self.ffn = max(32, int(round(8 * self.d / 3 / 32)) * 32)
        assert self.d % self.n_heads == 0
        assert (self.d // self.n_heads) % 2 == 0, "RoPE necesita head_dim par"
        assert 0 < self.rope_frac <= 1 and self.rotary_dim % 2 == 0, "rope_frac·head_dim debe ser par"
        assert not self.rope_factors or len(self.rope_factors) == self.rotary_dim // 2, "rope_factors: uno por frecuencia"
        assert self.n_core == 0 or self.n_blocks >= 1

    # --- derivados -------------------------------------------------------
    @property
    def head_dim(self) -> int:
        return self.d // self.n_heads

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.rope_frac) // 2 * 2

    @property
    def recurrent(self) -> bool:
        return self.n_core > 0 and self.n_blocks == 1

    @property
    def scale_stack(self) -> float:
        """1/√(2L) para las ramas residuales de preludio y coda."""
        if self.res_scale > 0:
            return self.res_scale
        L = self.n_pre + self.n_coda
        return 1.0 / math.sqrt(2 * L) if L > 0 else 1.0

    @property
    def scale_core(self) -> float:
        """1/√(2L) dentro de un bloque de núcleo. El estado se renormaliza tras
        cada iteración, así que L es la profundidad del bloque, no la desenrollada."""
        return 1.0 / math.sqrt(2 * self.n_core) if self.n_core > 0 else 1.0

    @property
    def logit_scale(self) -> float:
        """Pesos atados: emb ~ N(0,1) entra con rms 1 y los logits salen con std ~1."""
        return self.logit_scale_fixed if self.logit_scale_fixed > 0 else 1.0 / math.sqrt(self.d)

    def es_rejilla(self, i: int) -> bool:
        """¿La capa i-ésima de la pila (preludio y coda numeradas seguidas) usa la rejilla?"""
        return self.grid_every > 0 and i % self.grid_every != 0

    def n_params(self) -> int:
        d, f = self.d, self.ffn
        layer = 4 * d * d + 3 * d * f + 2 * d
        n_layers = self.n_pre + self.n_coda + self.n_core * self.n_blocks
        return self.vocab * d + self.abacus * d + n_layers * layer + d

    def layer_apps(self, r: int) -> int:
        """Aplicaciones de capa en un forward con r iteraciones (unidad de cómputo)."""
        return self.n_pre + self.n_coda + self.n_core * r

    def to_dict(self) -> dict:
        return asdict(self)
