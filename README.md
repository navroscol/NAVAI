# NAVROS — Neural Adaptive Vector Recurrence Orchestration System

Un modelo de lenguaje y un razonador algorítmico entrenables desde cero en hardware
modesto (Kaggle: 2×T4 o TPU v5e-8), escritos dos veces:

- **`navros/oracle/`** — NumPy puro, backward escrito a mano, verificado contra diferencias
  finitas. Corre en Windows sin CUDA y sirve de oráculo.
- **`navros/pt/`** — PyTorch, verificado contra el oráculo (mismos pesos, misma entrada:
  pérdida, todos los gradientes y la trayectoria de entrenamiento).

> **Estado: en construcción.** Las verificaciones de corrección están hechas y pasan. Los
> primeros experimentos (razonador y optimizadores) ya tienen resultados, más abajo, incluidos
> los que contradicen hipótesis de partida. El modelo de lenguaje de 1B aún no está entrenado.

## Resultados medidos (Kaggle, 2×T4; 3 semillas; media ± desv. típica)

Todos los números salen de `results/kaggle/`. El learning rate se barre para cada modelo y se
elige solo con validación en distribución; la prueba de longitud no se toca hasta el final.

### Razonador: bucle recurrente contra referencia fija de mismo cómputo

Entrenado con 8 dígitos / 8 operadores. «Bucle»: 1 capa iterada r̄=8 veces (~211K parámetros).
«Fijo»: 8 capas distintas con el mismo grafo (~1,6M parámetros, 7,7×), mismo cómputo de forward.

**Suma, sin RoPE (solo ábaco)** — exactitud de secuencia completa:

| longitud | bucle | fijo | bucle entrenado con r fijo |
|---|---|---|---|
| 8 (1×) | 100,0 ± 0,0 | 99,7 ± 0,2 | 100,0 ± 0,1 |
| 16 (2×) | 83,0 ± 9,2 | **94,6 ± 1,7** | 91,6 ± 7,4 |
| 24 (3×) | 75,5 ± 12,1 | **81,8 ± 2,9** | 83,1 ± 9,2 |
| 32 (4×) | 60,2 ± 7,6 | **67,8 ± 1,9** | 70,6 ± 9,5 |

**Con RoPE + ábaco** (primera versión), los dos se derrumban: 10,2 % (bucle) y 5,6 % (fijo) a 2×.
Lo que da la generalización de longitud en la suma es el ábaco sin RoPE, **no** la recurrencia.

**Expresiones mod 10 con RoPE** — en distribución (8 operadores), secuencia completa:
bucle **93,9 ± 2,2** contra fijo 46,5 ± 6,9, con 7,6× menos parámetros y 0,65× el cómputo de
entrenamiento. Es el resultado más favorable a la recurrencia. Fuera de distribución, ambos ≈ 0.

**r aleatorio frente a r fijo en entrenamiento** (expresiones, 8 operadores, secuencia según r):

| r en evaluación | 4 | 6 | 8 | 16 | 64 |
|---|---|---|---|---|---|
| entrenado con r fijo = 8 | 0,8 | 20,5 | **75,7** | 25,1 | 20,2 |
| entrenado con r aleatorio (r̄ = 8) | 33,2 | 84,1 | **93,9** | 91,5 | 90,4 |

Con r fijo, el modelo aprende «aplica exactamente 8 pasos». Con r aleatorio, iterar más no
lo rompe. Pero **tampoco lo mejora**: en ningún caso medido, pensar más allá de r̄ aumentó la
exactitud fuera de distribución.

**Ancho del bucle** (suma sin RoPE, secuencia completa):

| ancho | parámetros | 2× | 3× | 4× |
|---|---|---|---|---|
| 64 | 52K | 16,4 | 0,6 | 0,0 |
| 128 | 211K | 90,8 | 78,0 | 61,2 |
| 256 | 799K | **98,3** | **86,0** | **81,8** |

Aquí, más ancho generaliza mejor, no peor.

### Muon contra AdamW (modelo de bytes sobre FineWeb-Edu, mejor contra mejor)

| ancho | pasos | AdamW | Muon | pasos que necesita Muon para igualar el final de AdamW |
|---|---|---|---|---|
| 128 | 250 | 2,071 ± 0,035 | **1,778 ± 0,008** | 120 (2,1×) |
| 128 | 1000 | 1,539 ± 0,004 | **1,490 ± 0,002** | 750 (1,3×) |
| 256 | 250 | 2,002 ± 0,025 | **1,642 ± 0,009** | 108 (2,3×) |
| 256 | 1000 | 1,492 ± 0,005 | **1,417 ± 0,008** | 650 (1,5×) |
| 512 | 250 | 1,938 ± 0,026 | **1,586 ± 0,003** | 108 (2,3×) |
| 512 | 1000 | 1,458 ± 0,007 | **1,371 ± 0,001** | 600 (1,7×) |

Muon gana en todos los anchos, también en C=128. La ventaja crece con el ancho y **se reduce
con el horizonte**. El coste por paso no se midió de forma válida (dos procesos compartían GPU).

### LM pequeño: recurrente contra pilas fijas (Modal, 1×H100; **1 semilla**)

Corpus bilingüe propio (FineWeb2-HQ es + FineWeb-Edu en, tokenizador BPE de 32K), d=512,
100M tokens por modelo, mismo entrenador que el 1B (bf16, Muon + AdamW, WSD). Datos en
`results/modal/lm_estudio/`.

| modelo | parámetros | GFLOP/token (entreno) | test es | test en | tiempo |
|---|---|---|---|---|---|
| rec-s: 2 + núcleo de 4 iterado (r̄=4, k=2) + 2 | 42,1M | 0,43 | 4,123 | 4,313 | 394 s |
| fix20-s: 20 capas distintas | 80,0M | 0,54 (1,28×) | **4,019** | **4,199** | 557 s |
| fix8-s: 8 capas (mismos parámetros que rec-s) | 42,1M | 0,28 (0,65×) | 4,134 | 4,324 | 278 s |

Pérdida de validación del recurrente según r en evaluación:

| idioma | r=1 | r=2 | r=3 | r=4 | r=8 | r=16 |
|---|---|---|---|---|---|---|
| es | 4,0263 | 4,0199 | 4,0195 | 4,0195 | 4,0196 | 4,0196 |
| en | 4,4552 | 4,4457 | 4,4449 | 4,4448 | 4,4447 | 4,4447 |

**En modelado de lenguaje a esta escala, la recurrencia casi no se usa.** Las vueltas después de
la primera aportan ≤ 0,011 nats, y a partir de r=3 la curva es plana (punto fijo). El recurrente
gana a fix8-s por solo 0,011 nats, con 1,5× el cómputo, dentro de lo que una semilla no permite
distinguir. fix20-s gana al recurrente por 0,10–0,11 nats con los mismos tokens.

Lo que este estudio **no** demuestra:
- No es de igual cómputo estricto: fix20-s gastó un 28 % más de FLOPs de entrenamiento.
- El LR no quedó acotado: solo se barrieron 0,01 y 0,02, y los tres eligieron el borde (0,02).
- 100M tokens es ≈ 0,1× Chinchilla. Una sola semilla.
- Mide perplejidad, no razonamiento. En el razonador, la recurrencia sí ganaba en expresiones
  en distribución.

## Arquitectura

Un solo modelo cubre todos los casos (`navros/config.py`):

```
preludio:  x0 = emb[tokens] (+ ábaco[pos]) → capas de preludio
núcleo:    h ← rmsnorm_sin_parámetros( bloque( h + x0 ) )      r veces
coda:      capas de coda → logits = rmsnorm(h) @ embᵀ / √d     (pesos atados)
```

- RMSNorm, RoPE, SwiGLU, pesos atados, sin sesgos, ramas residuales escaladas por 1/√(2L).
- **Razonador recurrente**: mismos pesos en cada iteración, inyección de entrada `+ x0`,
  norma sin parámetros en cada vuelta, r aleatorio (Poisson-lognormal) en entrenamiento,
  BPTT truncado por las últimas k iteraciones.
- **Referencia fija**: el mismo grafo con D bloques de pesos distintos (D = r̄): mismo
  cómputo de forward que el bucle a su r medio.
- **Optimizador**: Muon (Newton-Schulz) para matrices; AdamW para `emb`, `abaco` y ganancias.

## Verificación (reproducible en cualquier CPU)

```bash
python scripts/01_gradcheck.py      # oráculo contra diferencias finitas
python scripts/02_verify_port.py    # PyTorch contra el oráculo
python -m pytest -q tests           # incluye controles negativos: bugs inyectados que deben detectarse
```

| prueba | resultado | criterio |
|---|---|---|
| gradientes vs diferencias finitas (derivada direccional por tensor, barrido de eps), 6 configuraciones | peor error 2,1·10⁻⁸ | < 10⁻⁴ |
| el error cae como eps² en eps grande | pendiente 2,00 | truncamiento, no bug |
| port: gradientes en float64 | ~10⁻¹⁵ | < 2·10⁻⁵ |
| port: gradientes en float32 | ~10⁻⁶ | < 2·10⁻⁵ |
| port: pesos tras 20 pasos de Muon / AdamW | ~10⁻⁸ | < 2·10⁻⁵ |
| controles negativos (bug en RMSNorm, en RoPE, pesos alterados, etiquetas corruptas) | todos detectados | — |

## Tareas verificables

- **suma** (`navros/tasks/suma.py`): n dígitos, respuesta completa en una pasada, sin tokens
  intermedios. Etiquetas comprobadas con la aritmética entera de Python.
- **expr** (`navros/tasks/expr.py`): expresiones módulo 10 con precedencia de Python,
  supervisando cada subexpresión. Etiquetas recalculadas con `ast` + `eval()` sobre el nodo
  que Python asigna a cada posición.

Se entrena con longitud L y se evalúa en L, 2L, 3L, 4L, con dos métricas: secuencia completa
y posición a posición.

## Estructura

```
navros/config.py        configuración compartida
navros/oracle/          NumPy: capas, modelo, Muon/AdamW, gradcheck
navros/pt/              PyTorch: modelo, Muon/AdamW
navros/tasks/           suma, expr, pool pregenerado y verificado
navros/reasoner.py      entrenamiento/evaluación bucle vs fijo
navros/lm_sweep.py      Muon vs AdamW, mejor contra mejor
navros/experiments.py   estudios completos (rejillas repartidas entre GPUs)
kaggle/                 scripts de control de los notebooks de Kaggle
```
