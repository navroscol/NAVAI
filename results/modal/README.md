# NAVROS-1B en Modal (H100 80 GB) — benchmark, 2026-09-19

Entrenador `navros/lm_ddp.py` en un proceso: modelo bf16, maestros fp32, momento de Muon fp32,
sin recomputación de activaciones. T=1024, 64 secuencias por paso, micro-lote 8, tokens sintéticos.

| modelo | tokens/s (pasos 3–8) | TFLOP/s útiles | memoria máx. | coste por 1.000 M tokens (3,95 $/h) |
|---|---|---|---|---|
| navros-1b (recurrente, r̄=4, k=2; 11,05 GFLOP/token) | ≈ 31.800 (24.700–49.100 según r) | ≈ 352 (≈ 36 % del pico bf16) | 38,8 GB | ≈ 34 $ |
| navros-1b-fix (18 capas; 6,52 GFLOP/token) | ≈ 49.100 | ≈ 320 | 38,8 GB | ≈ 22 $ |

Comparación: Kaggle 2×T4 da ≈ 2.200 (recurrente) y ≈ 3.150 (fijo) tokens/s → la H100 rinde ≈ 15×.
Registro completo en `bench_h100.txt`.
