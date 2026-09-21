# Proyección: escalar NAVROS de 2B a 40B parámetros

Calibración: Chinchilla predice 2.987 nats para el 1B con 2.600M tokens; medido 3.040. Desplazamiento aplicado: +0.053 nats.

Supuestos: H100 en Modal a 4,42 $/h (tarifa implícita en el README), 320 TFLOP/s útiles y 49K tok/s medidos con el 1B; 85 % de eficiencia al repartir entre GPUs; 16 bytes por parámetro en memoria (sin ZeRO); tokens = 20 × parámetros (Chinchilla). Pérdida en nats/token de prueba.

## Coste y tiempo a tokens óptimos (D = 20 N)

| parámetros | tokens | FLOP | horas de H100 | coste | memoria de entreno | GPUs mínimas (memoria) | días con 8×H100 | pérdida proyectada | perplejidad |
|---|---|---|---|---|---|---|---|---|---|
| 1B | 20B | 1.2e+20 | 104 | 541 $ | 16 GB | 1 | 0.6 | 2.63 | 13.9 |
| 2B | 40B | 4.8e+20 | 417 | 2.166 $ | 32 GB | 1 | 2.6 | 2.46 | 11.8 |
| 4B | 80B | 1.9e+21 | 1.667 | 8.663 $ | 64 GB | 1 | 10.2 | 2.33 | 10.3 |
| 7B | 140B | 5.9e+21 | 5.104 | 26.531 $ | 112 GB | 2 | 31.3 | 2.24 | 9.4 |
| 13B | 260B | 2.0e+22 | 17.604 | 91.504 $ | 208 GB | 3 | 107.9 | 2.15 | 8.6 |
| 20B | 400B | 4.8e+22 | 41.667 | 216.578 $ | 320 GB | 5 | 255.3 | 2.10 | 8.2 |
| 40B | 800B | 1.9e+23 | 166.667 | 866.310 $ | 640 GB | 10 | 817.0 | 2.03 | 7.7 |

## Lo mismo con el presupuesto de tokens que ya se ha usado (2.600M) y con 10×

| parámetros | tokens | horas de H100 | coste | pérdida proyectada |
|---|---|---|---|---|
| 1B | 2.6B | 14 | 70 $ | 3.05 |
| 1B | 26.0B | 135 | 704 $ | 2.60 |
| 2B | 2.6B | 27 | 141 $ | 2.97 |
| 2B | 26.0B | 271 | 1.408 $ | 2.52 |
| 7B | 2.6B | 95 | 493 $ | 2.87 |
| 7B | 26.0B | 948 | 4.927 $ | 2.42 |
| 40B | 2.6B | 542 | 2.816 $ | 2.79 |
| 40B | 26.0B | 5.417 | 28.155 $ | 2.34 |

## Qué da cada dólar

- 2B a tokens óptimos: -0.58 nats respecto al 1B actual, por 29× el coste del último tramo de 2.600M tokens.
- 7B a tokens óptimos: -0.80 nats respecto al 1B actual, por 359× el coste del último tramo de 2.600M tokens.
- 40B a tokens óptimos: -1.01 nats respecto al 1B actual, por 11737× el coste del último tramo de 2.600M tokens.

Kaggle (2×T4, 30 h/semana): el 1B con 950M tokens costó 5,5 h de H100, que son unas 60 a 80 h de T4; 2B a tokens óptimos serían del orden de 3.000 h de T4. No es una opción para nada por encima de lo actual.
