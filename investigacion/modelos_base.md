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
| **Motif-2.6B v1.1-LC** (Motif Technologies) | 2,6B | MIT (la v1.0 es Apache 2.0) | por comprobar (entrenado desde cero, inglés y coreano principalmente) | Differential Attention y PolyNorm en vez de atención y RMSNorm estándar | alto: hay que implementar las dos capas en `navros/pt/model.py` | MIT y tamaño ideal, pero arquitectura distinta |
| **MiMo-7B-Base** (Xiaomi) | 7B | MIT | por comprobar (25T tokens, orientado a razonamiento) | 36 capas, d=4096, FFN 11008, Llama-like con cabezas MTP extra | medio: quitar MTP, expandir KV si hay GQA | MIT, pero 7B: maestros fp32 no caben en una H100 (112 GB); haría falta bf16 o 2 GPUs |
| DeepSeek-R1-Distill-Qwen-7B | 7B | MIT (base Qwen2.5) | sí | Qwen2 (sesgos QKV, GQA) | medio | igual que el de 1,5B, pero 7B |
| Seed-Coder-8B-Base (ByteDance) | 8B | MIT | código | Llama-like | bajo-medio | solo código; 8B |
| GLM-4-9B-0414 (Zhipu) | 9B | MIT | sí | GLM (atención y norma propias) | alto | MIT, pero 9B y arquitectura propia |
| Phi-4 (Microsoft) | 14B | MIT | sí | Phi-3 grande, sin GQA | bajo | no cabe en una H100 con maestros fp32 (224 GB) |
| Qwen3-1.7B-Base | 1,7B | Apache 2.0 | sí | d=2048, 16 cabezas, FFN 6144 (**la misma forma que navros-1b-fix**), 28 capas, GQA 8, QK-norm, vocab 152K | medio: QK-norm y expandir KV | el mejor por forma y tamaño, pero no es MIT |
| SmolLM2-1.7B | 1,7B | Apache 2.0 | flojo (inglés) | d=2048, 24 capas, 32 cabezas sin GQA, vocab 49K, atados | muy bajo | porte casi directo; le falta español |
| Gemma 4 E2B / E4B (Google) | 2,3B / 4,5B efectivos | Apache 2.0 (Gemma 4; las versiones anteriores tenían términos propios) | sí | atención local/global alternada, embeddings por capa, KV compartido | alto: arquitectura distinta | no portable sin reescribir el modelo |
| Devstral Small 2 (Mistral) | 24B | Apache 2.0 | sí | Mistral | — | demasiado grande para esta cadena |
| Devstral 2 (Mistral) | 123B | MIT modificada (sin uso si la empresa factura > 20 M$/mes) | sí | — | — | demasiado grande |
| Codestral 22B v0.1 (Mistral) | 22B | **Mistral Non-Production**: sin uso comercial | — | — | — | no; Codestral 2 (2026) sí es Apache 2.0, pero 22B |
| Llama 3.2 1B/3B (Meta) | 1B / 3B | Licencia Llama (no MIT) | sí | GQA | medio | licencia propia, no permisiva del todo |
| Youtu-LLM-2B-Base (Tencent) | 2B | licencia propia "youtu-llm" | — | — | — | descartado por licencia |

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

## Resumen de lo estrictamente MIT y ≤ 4B (lo que cabe en la cadena de una H100)

1. **Phi-3.5-mini-instruct**, 3,8B: el único con licencia MIT, español y la misma forma de bloque
   que NAVROS (sin GQA, RoPE completo, SwiGLU, sin sesgos, vocab 32K). Es un modelo de instrucción,
   no una base limpia; el preentrenamiento continuado lo tolera.
2. **Phi-4-mini-instruct / Phi-4-mini-reasoning**, 3,8B: MIT, español, más moderno; GQA, RoPE parcial
   y vocab de 200K exigen tres adaptaciones en el código.
3. **DeepSeek-R1-Distill-Qwen-1.5B**: MIT, la única opción MIT por debajo de 2B; sesgos QKV y GQA.
4. **Motif-2.6B v1.1-LC**: MIT y desde cero, pero con atención y normalización no estándar.

## Fuentes

- Phi-4-mini-instruct, MIT, idiomas: https://huggingface.co/microsoft/Phi-4-mini-instruct
- Phi-3.5-mini-instruct: https://huggingface.co/microsoft/Phi-3.5-mini-instruct y su `config.json` (32 cabezas, 32 KV, vocab 32.064, embeddings no atados): https://huggingface.co/unsloth/Phi-3-mini-4k-instruct/blob/79515e10301621a883bebe7e63693c72012744a5/config.json
- DeepSeek-R1-Distill-Qwen-1.5B, MIT: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
- Gemma 4 bajo Apache 2.0: https://www.ghacks.net/2026/04/06/google-releases-gemma-4-in-four-model-sizes-under-apache-2-0-license/ y https://huggingface.co/blog/gemma4 ; términos anteriores: https://ai.google.dev/gemma/terms
- Codestral 22B (MNPL): https://mistral.ai/news/codestral/ ; Codestral 2 Apache 2.0: https://aitooltier.com/tools/codestral
- Devstral 2 (MIT modificada) y Devstral Small 2 (Apache 2.0): https://huggingface.co/mistralai/Devstral-2-123B-Instruct-2512 , https://simonwillison.net/2025/Dec/9/devstral-2/
- Qwen3-1.7B-Base, Apache 2.0: https://huggingface.co/Qwen/Qwen3-1.7B-Base
- Phi-4-mini-reasoning, MIT, misma arquitectura que Phi-4-mini (vocab 200K, GQA, embeddings compartidos): https://huggingface.co/microsoft/Phi-4-mini-reasoning , https://arxiv.org/pdf/2503.01743
- Motif-2.6B (Apache 2.0) y v1.1-LC (MIT); arquitectura con Differential Attention y PolyNorm: https://huggingface.co/Motif-Technologies/Motif-2.6B , https://huggingface.co/Motif-Technologies/Motif-2.6b-v1.1-LC , https://arxiv.org/abs/2508.09148
- MiMo-7B-Base, MIT, 36 capas, d=4096: https://huggingface.co/XiaomiMiMo/MiMo-7B-Base , https://github.com/XiaomiMiMo/MiMo
- Seed-Coder-8B-Base, MIT: https://huggingface.co/ByteDance-Seed/Seed-Coder-8B-Base
- GLM-4-9B-0414, MIT: https://huggingface.co/zai-org/GLM-4-9B-0414
- Youtu-LLM-2B-Base, licencia propia: https://huggingface.co/tencent/Youtu-LLM-2B-Base
