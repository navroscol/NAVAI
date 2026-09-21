# Cómo bajar más la pérdida del 1B a igual cómputo: candidatos, coste y lo medido

Objetivo: dado que los 2.000 $ en tramos de 30 $ compran ~52.000M tokens (pérdida proyectada
3,04 → 2,50), ¿qué se puede cambiar en el entrenamiento para que esa curva baje más deprisa o más
abajo, sin subir el coste? Aquí se ordenan los candidatos por lo que prometen, lo que cuestan y lo
que se ha podido comprobar con el entrenador real (`navros/lm_ddp.py`) en esta CPU.

Todo lo citado es de memoria y hay que verificarlo con fuentes propias antes de apoyarse en ello.

## 1. Los candidatos, ordenados

| # | técnica | qué cambia | ganancia esperada (literatura) | coste en la cadena | cómo se prueba |
|---|---|---|---|---|---|
| 1 | **Weight decay en Muon** | `wd` de 0 a ~0,1 en las matrices (`LMRun.wd`, ya soportado en `navros/pt/optim.py`) | Liu et al. 2025 ("Muon is scalable"): sin wd las normas crecen sin freno en corridas largas y la pérdida final es peor; con wd la ventaja de Muon sobre AdamW se mantiene a 1B+. Efecto pequeño a pocos tokens, grande a 50.000M | cero | A/B aquí (§3) y un tramo de 30 $ |
| 2 | **Promedio de checkpoints en vez de decaer** | Media de los últimos N checkpoints de la fase estable | Hägele et al. 2024 ("Scaling laws and compute-optimal training beyond fixed training durations"): el promedio (SWA/EMA) recupera casi toda la ganancia del decaimiento **sin gastarlo**, y permite evaluar "como si hubiera decaído" en cualquier punto de la fase estable | cero: los checkpoints ya se guardan | A/B aquí (§3); en la cadena, un `promedio.py` sobre los exports de GCS |
| 3 | **Ajustar el LR de la fase larga** | El barrido del 1B solo probó 0,01 y 0,02 y eligió el borde | Un LR mal puesto en 50.000M tokens cuesta más que cualquier truco. Muon con wd tolera LR más alto | un tramo por valor probado | A/B aquí con 3 valores (§3); en la cadena, decidirlo en el primer tramo |
| 4 | **Contexto más largo (T=2048)** | `T` de 1024 a 2048 con la mitad de secuencias por paso | La pérdida por token baja al tener más contexto (0,02 a 0,05 nats en texto web según se cite); además el modelo sirve para textos más largos. Atención cuadrática: ~10 % menos tokens/s a 1B | ~10 % de tokens | A/B aquí con T 256→512 (§3); en la cadena, fijarlo desde el primer tramo (no se puede cambiar a mitad sin reajuste) |
| 5 | **Predicción de varios tokens (MTP)** | Cabeza extra que predice t+2 (y t+3) además de t+1 | Gloeckle et al. 2024: mejora clara en código y en modelos ≥ 1B, poco en texto a 1B; DeepSeek-V3 la usa. Coste ~10 % de cómputo | ~10 % de tokens y tocar el modelo (`navros/pt/model.py`) | no aquí: requiere cambiar el modelo; un tramo de 30 $ |
| 6 | **Mezcla de datos y calidad** | Más peso a FineWeb-Edu de puntuación alta; deduplicar entre particiones | Es el factor con más efecto en la literatura (FineWeb-Edu contra FineWeb: 2 a 4 puntos en pruebas), pero tu corpus ya es Edu; lo que queda es afinar la mezcla es/en y la repetición | cero | no medible aquí (no hay texto); en la cadena, decidir la mezcla antes de congelar el corpus |
| 7 | **Destilación de un modelo mayor** | Entrenar contra los logits de un maestro de 7B en vez de contra el token | Gemma 2: a igual tokens, destilar da la calidad de "muchos más tokens". Es la ganancia más grande conocida para modelos pequeños | **inviable aquí**: el maestro tendría que usar tu tokenizador de 32K propio (ningún modelo público lo usa) y su forward costaría 2× el entrenamiento | descartada |
| 8 | **Mezcla de expertos (MoE)** | 1B activos, 4 a 8B totales | Menor pérdida a igual FLOPs de entreno (DeepSeekMoE) | memoria 4 a 8×: no cabe en una H100 con maestros fp32 y Adam sin ZeRO; tocaría reescribir el entrenador | descartada para esta cadena |
| 9 | **Rejilla / MPS en el bloque** | Sustituir atención por estado matricial | Sin evidencia de menor pérdida (ver `investigacion/superposicion/INFORME.md` §11.3); paridad en el mejor caso | tocar el modelo | la prueba de 20 $ de dos modelos de 100M, no la cadena |

Lo que no está en la tabla porque ya lo tienes: Muon (medido 1,3 a 2,3× sobre AdamW), WSD, RoPE,
SwiGLU, pesos atados, bf16 con maestros fp32.

## 2. Qué se puede comprobar aquí y qué no

En esta máquina no hay texto (la red no llega a Hugging Face), así que el A/B usa el único corpus
grande disponible: el código fuente de la biblioteca estándar de Python (113 MB, 58M tokens con tu
tokenizador, reducido a un vocabulario de 4.096 para que un modelo de 1,4M parámetros entrene a
11.700 tokens/s en 4 núcleos). Es código, no lenguaje; pero las técnicas 1 a 4 actúan sobre el
optimizador, el calendario y la longitud, no sobre el contenido, y eso es lo que se mide.

Modelo: 4 capas, d = 128, la misma pila fija que `navros-1b-fix` en pequeño; mismo entrenador,
mismo Muon, mismo WSD (calentamiento 50 pasos, decaimiento 20 %), 6M tokens por variante (unos 4
tokens por parámetro: poco, pero es lo que cabe en una hora). Una semilla; la segunda si da tiempo.

Lo que **no** prueba: ninguna cifra de aquí se traslada al 1B; lo único trasladable es el signo y
el orden de los efectos.

## 3. Resultados del A/B

PENDIENTE_AB

## 4. Recomendación para la cadena

PENDIENTE_RECOMENDACION
