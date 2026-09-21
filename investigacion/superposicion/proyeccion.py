"""Proyección de coste, tiempo y pérdida para escalar NAVROS de 2B a 40B parámetros.

No mide nada nuevo: extrapola a partir de (a) lo medido en el README del proyecto para el 1B en
Modal (H100: 49K tok/s, 320 TFLOP/s útiles, 24,3 $ por 5,5 h) y (b) la ley de escala de Hoffmann
et al. 2022 (Chinchilla), L(N, D) = E + A/N^α + B/D^β, calibrada con un desplazamiento para que
reproduzca la pérdida medida del 1B con 2.600M tokens (prueba es 3,036 / en 3,045).

    python proyeccion.py > resultados/proyeccion.md
"""
from __future__ import annotations

import math

# --- medido en este proyecto (README) ---
N_1B = 1_048_651_776
TOK_S_H100 = 49_000          # tokens/s por H100 con T=1024, bf16, Muon+AdamW
TFLOPS_UTIL = 320e12         # FLOP/s útiles medidos
USD_H = 24.3 / 5.5           # 4,42 $/h de H100 en Modal según el README
PERDIDA_1B_2600M = (3.036 + 3.045) / 2
TOKENS_1B = 2.6e9

# --- Chinchilla (Hoffmann et al. 2022, ajuste 3): a verificar con fuentes propias ---
E, A, ALPHA, B, BETA = 1.69, 406.4, 0.34, 410.7, 0.28


def chinchilla(N, D):
    return E + A / N ** ALPHA + B / D ** BETA


DESPL = PERDIDA_1B_2600M - chinchilla(N_1B, TOKENS_1B)  # calibración con el dato propio


def fila(N, D, gpus, eficiencia=0.85):
    flops = 6 * N * D
    horas_h100 = flops / TFLOPS_UTIL / 3600
    mem_gb = 16 * N / 1e9                       # bf16 pesos + maestros fp32 + Adam (16 B/parámetro)
    gpus_min = max(1, math.ceil(mem_gb / 70))    # sin ZeRO ni offload: 70 GB útiles por H100
    g = max(gpus, gpus_min)
    dias = horas_h100 / (g * eficiencia) / 24
    return dict(N=N, D=D, flops=flops, horas_h100=horas_h100, usd=horas_h100 * USD_H / eficiencia,
                mem_gb=mem_gb, gpus_min=gpus_min, gpus=g, dias=dias, perdida=chinchilla(N, D) + DESPL)


def fmt(x):
    return f"{x:,.0f}".replace(",", ".")


if __name__ == "__main__":
    print("# Proyección: escalar NAVROS de 2B a 40B parámetros\n")
    print(f"Calibración: Chinchilla predice {chinchilla(N_1B, TOKENS_1B):.3f} nats para el 1B con 2.600M tokens; "
          f"medido {PERDIDA_1B_2600M:.3f}. Desplazamiento aplicado: {DESPL:+.3f} nats.\n")
    print("Supuestos: H100 en Modal a 4,42 $/h (tarifa implícita en el README), 320 TFLOP/s útiles y "
          "49K tok/s medidos con el 1B; 85 % de eficiencia al repartir entre GPUs; 16 bytes por parámetro "
          "en memoria (sin ZeRO); tokens = 20 × parámetros (Chinchilla). Pérdida en nats/token de prueba.\n")
    print("## Coste y tiempo a tokens óptimos (D = 20 N)\n")
    print("| parámetros | tokens | FLOP | horas de H100 | coste | memoria de entreno | GPUs mínimas (memoria) | días con 8×H100 | pérdida proyectada | perplejidad |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for N in (1e9, 2e9, 4e9, 7e9, 13e9, 20e9, 40e9):
        r = fila(N, 20 * N, gpus=8)
        print(f"| {N/1e9:.0f}B | {r['D']/1e9:.0f}B | {r['flops']:.1e} | {fmt(r['horas_h100'])} | {fmt(r['usd'])} $ | "
              f"{r['mem_gb']:.0f} GB | {r['gpus_min']} | {r['dias']:.1f} | {r['perdida']:.2f} | {math.exp(r['perdida']):.1f} |")
    print("\n## Lo mismo con el presupuesto de tokens que ya se ha usado (2.600M) y con 10×\n")
    print("| parámetros | tokens | horas de H100 | coste | pérdida proyectada |")
    print("|---|---|---|---|---|")
    for N in (1e9, 2e9, 7e9, 40e9):
        for D in (2.6e9, 26e9):
            r = fila(N, D, gpus=8)
            print(f"| {N/1e9:.0f}B | {D/1e9:.1f}B | {fmt(r['horas_h100'])} | {fmt(r['usd'])} $ | {r['perdida']:.2f} |")
    print("\n## Qué da cada dólar\n")
    base = fila(N_1B, TOKENS_1B, 1)
    for N in (2e9, 7e9, 40e9):
        r = fila(N, 20 * N, 8)
        print(f"- {N/1e9:.0f}B a tokens óptimos: {r['perdida'] - base['perdida']:+.2f} nats respecto al 1B actual, "
              f"por {r['usd'] / base['usd']:.0f}× el coste del último tramo de 2.600M tokens.")
    print("\nKaggle (2×T4, 30 h/semana): el 1B con 950M tokens costó 5,5 h de H100, que son unas 60 a 80 h de T4; "
          "2B a tokens óptimos serían del orden de 3.000 h de T4. No es una opción para nada por encima de lo actual.")
