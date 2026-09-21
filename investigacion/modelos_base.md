# Modelos base abiertos "compatibles con NAVROS": licencias y portabilidad

Qué significa aquí "compatible": que sus pesos se puedan **cargar en el código de NAVROS**
(`navros/pt/model.py`: pila pre-norma con RMSNorm, RoPE completo, SwiGLU, sin sesgos, pesos atados,
atención multi-cabeza sin GQA) y seguir entrenándolos con Muon y la cadena de tramos, en vez de
partir de cero. Fusionar pesos entre familias distintas no es posible; lo que sí es posible es
**arrancar** de un modelo ya entrenado, con su tokenizador, y continuar el preentrenamiento con
nuestro corpus y nuestro entrenador.

Licencias verificadas en la web el 21-9-2026 (fuentes al final); las formas de las arquitecturas
son de memoria donde no se pudo leer el `config.json` (la red de esta sesión no llega a Hugging
Face) y hay que comprobarlas antes de portar.

| modelo | parámetros | licencia | español | forma respecto a NAVROS | esfuerzo de porte | veredicto |
|---|---|---|---|---|---|---|
| **Phi-3.5-mini-instruct** (Microsoft) | 3,8B | MIT | sí (multilingüe) | d=3072, 32 capas, 32 cabezas **sin GQA**, RoPE completo, SwiGLU, sin sesgos, vocab 32.064; embeddings **no** atados | bajo: solo la cabeza de salida separada | **el más compatible con licencia MIT** |
| **Phi-4-mini-instruct** (Microsoft) | 3,8B | MIT | sí (23 idiomas) | d=3072, 32 capas, GQA (24/8), RoPE parcial, vocab 200K | medio: expandir KV, RoPE parcial, embedding de 600M | viable, más trabajo |
| **DeepSeek-R1-Distill-Qwen-1.5B** | 1,5B | MIT (base Qwen2.5, Apache 2.0) | sí (Qwen2.5 es multilingüe) | Qwen2: GQA (12/2), **sesgos en QKV**, vocab 152K | medio: añadir sesgos o absorberlos, expandir KV | viable; es un ajuste de razonamiento, no una base limpia |
| Phi-4 (Microsoft) | 14B | MIT | sí | Phi-3 grande, sin GQA | bajo | no cabe en una H100 con maestros fp32 (224 GB) |
| Qwen3-1.7B-Base | 1,7B | Apache 2.0 | sí | d=2048, 16 cabezas, FFN 6144 (**la misma forma que navros-1b-fix**), 28 capas, GQA 8, QK-norm, vocab 152K | medio: QK-norm y expandir KV | el mejor por forma y tamaño, pero no es MIT |
| SmolLM2-1.7B | 1,7B | Apache 2.0 | flojo (inglés) | d=2048, 24 capas, 32 cabezas sin GQA, vocab 49K, atados | muy bajo | porte casi directo; le falta español |
| Gemma 4 E2B / E4B (Google) | 2,3B / 4,5B efectivos | Apache 2.0 (Gemma 4; las versiones anteriores tenían términos propios) | sí | atención local/global alternada, embeddings por capa, KV compartido | alto: arquitectura distinta | no portable sin reescribir el modelo |
| Devstral Small 2 (Mistral) | 24B | Apache 2.0 | sí | Mistral | — | demasiado grande para esta cadena |
| Devstral 2 (Mistral) | 123B | MIT modificada (sin uso si la empresa factura > 20 M$/mes) | sí | — | — | demasiado grande |
| Codestral 22B v0.1 (Mistral) | 22B | **Mistral Non-Production**: sin uso comercial | — | — | — | no; Codestral 2 (2026) sí es Apache 2.0, pero 22B |
| Llama 3.2 1B/3B (Meta) | 1B / 3B | Licencia Llama (no MIT) | sí | GQA | medio | licencia propia, no permisiva del todo |

## Lo que cambia respecto a "partir de cero"

- **Tokenizador.** Al portar un modelo se hereda su tokenizador; el corpus hay que volver a
  tokenizarlo con él (los shards actuales están en el BPE de 32K propio). Con Phi-3.5-mini (32.064)
  el coste de vocabulario es parecido al actual; con Qwen o Phi-4-mini (150 a 200K) la embedding
  sola pesa 300 a 600M parámetros.
- **Presupuesto.** Un 3,8B en la cadena gasta 3,7× más por token que el 1B: los 52.000M tokens se
  convierten en ~14.000M, pero se parte de un modelo que ya vio billones. Con Chinchilla calibrada
  no se puede proyectar (la base no es nuestra); la única medida honesta es un tramo de prueba.
- **Identidad.** El resultado sería un derivado (por ejemplo "NAVROS, derivado de Phi-3.5-mini de
  Microsoft, MIT"); el README tendría que decirlo. El 1B actual es el único que es NAVROS desde cero.

## Recomendación

Si el objetivo es el mejor asistente por dólar: **Phi-3.5-mini-instruct** portado al código de
NAVROS (o, más simple aún, ajustado con LoRA en su propio código) con los datos de SFT actuales más
datos sintéticos generados con NIM. Un workspace de 30 $ para la prueba.

Si el objetivo es el modelo propio desde cero y la investigación de arquitectura: seguir con la
cadena del 1B y no mezclar.

## Fuentes

- Phi-4-mini-instruct, MIT, idiomas: https://huggingface.co/microsoft/Phi-4-mini-instruct
- Phi-3.5-mini-instruct: https://huggingface.co/microsoft/Phi-3.5-mini-instruct y su `config.json` (32 cabezas, 32 KV, vocab 32.064, embeddings no atados): https://huggingface.co/unsloth/Phi-3-mini-4k-instruct/blob/79515e10301621a883bebe7e63693c72012744a5/config.json
- DeepSeek-R1-Distill-Qwen-1.5B, MIT: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
- Gemma 4 bajo Apache 2.0: https://www.ghacks.net/2026/04/06/google-releases-gemma-4-in-four-model-sizes-under-apache-2-0-license/ y https://huggingface.co/blog/gemma4 ; términos anteriores: https://ai.google.dev/gemma/terms
- Codestral 22B (MNPL): https://mistral.ai/news/codestral/ ; Codestral 2 Apache 2.0: https://aitooltier.com/tools/codestral
- Devstral 2 (MIT modificada) y Devstral Small 2 (Apache 2.0): https://huggingface.co/mistralai/Devstral-2-123B-Instruct-2512 , https://simonwillison.net/2025/Dec/9/devstral-2/
- Qwen3-1.7B-Base, Apache 2.0: https://huggingface.co/Qwen/Qwen3-1.7B-Base
