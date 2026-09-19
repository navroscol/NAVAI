# NAVROS-1B en Kaggle T4×2 — benchmark (tokens sintéticos), 2026-09-19

Commit `3091fff` (antes `bfb0c96`, ver `results/commits_reescritos.tsv`). Entrenador `navros/lm_ddp.py`: modelo fp16 en cada GPU, maestros fp32 y estado de
Muon (bf16) repartidos por dueño, reducción del gradiente al dueño, recomputación de activaciones.
T=1024, lote de 64 secuencias (65.536 tokens) por paso, 6 pasos por configuración.

| modelo | micro | tokens/s (pasos 2–6) | TFLOP/s útiles (2 GPU) | memoria máx. por GPU |
|---|---|---|---|---|
| navros-1b (recurrente, r̄=4, k=2; 11,05 GFLOP/token) | 2 | 1.690–3.180 (media ≈ 2.200; varía con r) | ≈ 22 | 9,3 GB |
| navros-1b | 4 | 1.740–3.240 (≈ 2.200) | ≈ 22 | 10,3 GB |
| navros-1b-fix (18 capas; 6,52 GFLOP/token) | 2 | 3.060–3.240 (≈ 3.150) | ≈ 20,5 | 9,1 GB |
| navros-1b-fix | 4 | 3.110–3.300 (≈ 3.250) | ≈ 21 | 10,1 GB |

- La primera versión (pesos y gradientes fp32 replicados + DDP) se quedó sin memoria (14,0 de 14,56 GB).
- ~21 TFLOP/s útiles entre las dos T4 ≈ 16 % del pico fp16 (la recomputación añade ~33 % que no cuenta).
- La cuota de Kaggle descuenta 2 h por cada hora de T4×2: 30 h/semana ≈ 15 h reales →
  ≈ 120 M tokens/semana para el 1B recurrente y ≈ 170 M para el fijo.
- Pérdida inicial ≈ 38–41 nats con vocabulario 32.768: con pesos atados y emb ~ N(0,1), el estado
  apunta al embedding del propio token de entrada y su logit vale ≈ √d ≈ 45. Transitorio conocido de
  los pesos atados; baja en los primeros pasos.
