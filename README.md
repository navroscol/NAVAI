# NAVROS — Neural Adaptive Vector Recurrence Orchestration System

Un modelo de lenguaje y un razonador algorítmico entrenables desde cero en hardware
modesto (Kaggle: 2×T4 o TPU v5e-8), escritos dos veces:

- **`navros/oracle/`** — NumPy puro, backward escrito a mano, verificado contra diferencias
  finitas. Corre en Windows sin CUDA y sirve de oráculo.
- **`navros/pt/`** — PyTorch, verificado contra el oráculo (mismos pesos, misma entrada:
  pérdida, todos los gradientes y la trayectoria de entrenamiento).

> **Estado: en construcción.** Las verificaciones de corrección están hechas y pasan.
> Los experimentos (razonador y optimizadores) se ejecutan en Kaggle; sus resultados, y la
> sección *Lo que esto NO demuestra*, se añadirán aquí cuando existan. No hay aún ninguna
> afirmación de rendimiento.

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
