# Phi-4-mini con el tokenizador de NAVROS: qué se hizo y cómo se lanza

Decisión: partir del cuerpo de Phi-4-mini (Microsoft, MIT) conservando nuestro tokenizador de 32K,
el corpus ya tokenizado y la cadena de tramos. El resultado es un derivado: "NAVROS, cuerpo de
Phi-4-mini (MIT), embedding y preentrenamiento continuado propios". El README deberá decirlo.

## Qué se conserva y qué se sustituye

| pieza | Phi-4-mini | en NAVROS |
|---|---|---|
| cuerpo: 32 capas, d=3072, 24 cabezas, FFN 8192, SwiGLU, RMSNorm | 3,2B parámetros | se copian tal cual; equivalencia exacta verificada |
| atención: GQA con 8 cabezas KV | 8 KV | se repiten a 24 (exacto); pasan a entrenarse por separado |
| RoPE: parcial (96 de 128 dims) con LongRoPE (factores cortos, mscale 1,19) | sí | añadido al modelo: `rope_frac`, `rope_factors`, `rope_mscale` |
| escalas: sin escala en ramas residuales ni en logits | 1 y 1 | `res_scale=1`, `logit_scale_fixed=1` |
| embedding: 200.064 × 3072, compartida entrada/salida | 614M | **nueva** de 32.768 × 3072 (100M), inicializada por trasplante: cada token nuestro = media de las embeddings de Phi de los tokens en que Phi parte su texto |
| tokenizador | o200k (200K) | **el nuestro** (32K), sin cambios en el corpus |

Parámetros del modelo resultante: 3,72B (con las cabezas KV expandidas), preset `phi4mini`.

## Archivos

- `navros/porte_phi.py`: mapeo de pesos, trasplante de embedding, export, y `--autotest` (referencia Phi-3
  independiente: error 0 entre ambas implementaciones con GQA, RoPE parcial y LongRoPE).
- `navros/config.py`, `navros/pt/model.py`, `navros/generate.py`: RoPE parcial y LongRoPE, escalas
  configurables. Con los valores por defecto el comportamiento es el de siempre (las 20 pruebas del
  repositorio pasan). El oráculo NumPy no implementa RoPE parcial y lo dice con un assert.
- `navros/lm.py`: preset `phi4mini`.
- `cloud/modal_porte_phi.py`: descarga, verificación contra `transformers` y export a GCS.
- `cloud/modal_cadena.py`: `--preset`, `--tag`, `--base-gcs`, `--grad-ckpt`.

## Cómo se lanza (Modal)

```bash
# 1) porte, una vez (CPU, ~15 min, < 1 $); deja base/phi_4_mini_instruct_navros32k.pt en el bucket
modal run cloud/modal_porte_phi.py --modelo microsoft/Phi-4-mini-instruct
#    (o --modelo microsoft/Phi-4-mini-reasoning: misma arquitectura)

# 2) cadena con su propia etiqueta y preset (no mezclar con la del 1B)
export NAVROS_PRESET=phi4mini NAVROS_TAG=navros-phi4mini-cadena
modal run cloud/modal_cadena.py::estado
modal run --detach cloud/modal_cadena.py::tramo --base-gcs base/phi_4_mini_instruct_navros32k.pt --micro 4
#    tramos siguientes: modal run --detach cloud/modal_cadena.py::tramo --micro 4
```

Lo que hay que mirar en el primer tramo:

- La verificación del porte imprime `error máximo` contra transformers: debe ser < 1e-2 sobre logits
  de magnitud ~20 a 40. Si no, no seguir.
- La pérdida inicial. Con la embedding trasplantada el modelo NO empieza en la pérdida de Phi: la
  embedding es nueva. Lo esperable es una pérdida alta al principio (4 a 6 nats) que cae deprisa en
  el primer tramo, porque el cuerpo ya sabe; si a las 2 horas no está por debajo del 1B actual
  (3,04), algo falla en el trasplante o en el LR.
- Memoria: 3,72B con maestros fp32 y Muon son unos 45 GB fijos; con `--micro 4` y T=1024 las
  activaciones caben en la H100. Si hay OOM, `--grad-ckpt` (un 30 % más lento) o `--micro 2`.
- Velocidad: 6 × 3,7 GFLOP/token ≈ 22 GFLOP/token; a 320 TFLOP/s serían ~14K tokens/s, unos
  250M tokens por tramo de 5 h. Los 55 tramos dan ~14.000M tokens, no 52.000M: hay que poner
  `--total-tokens 14000000000` en el primer tramo para que el decaimiento caiga donde toca.

## La rejilla sobre el cuerpo de Phi-4-mini (experimental)

Preset `phi4mini-rejilla`: en 24 de las 32 capas la atención softmax se sustituye por **atención
lineal con estado matricial**, que es la rejilla del estudio de superposición llevada a la
secuencia: el estado de cada cabeza es la matriz S_t = Σ_{s≤t} φ(k_s)·v_sᵀ (128×128, una suma de
productos exteriores clave·valor) y la lectura es o_t = φ(q_t)·S_t / (φ(q_t)·z_t), con
φ(x) = elu(x)+1. Las 8 capas restantes (0, 4, 8, …, 28) conservan la softmax para la recuperación
exacta, como recomendaba §11.3 del informe. Se reutilizan las proyecciones Q, K, V y O de Phi tal
cual: no hay parámetros nuevos, cambia la operación.

- Entrenamiento: forma cuadrática enmascarada, mismo coste que la softmax a T=1024 (`atencion_rejilla`).
- Generación: estado S y z por capa en la caché, en vez de la lista de claves y valores; memoria
  constante con la longitud. `tests/test_rejilla.py` comprueba que la forma cuadrática es la
  recurrencia, que el estado previo equivale a la secuencia entera, que la caché coincide con el
  forward completo y la causalidad.
- Se lanza con la misma base exportada: `NAVROS_PRESET=phi4mini-rejilla NAVROS_TAG=navros-phi4mini-rejilla`
  y los mismos comandos de la cadena. Es otra cadena (otra etiqueta).

Qué esperar, dicho claro: al cambiar la operación de 24 capas, el cuerpo de Phi deja de "saber"
en esas capas hasta que el entrenamiento las reajuste; la pérdida inicial será mucho más alta que
en `phi4mini` y la pregunta del experimento es si en los mismos tramos la alcanza. La literatura
que convierte modelos softmax en lineales (LoLCATs, "Mamba in the Llama", a verificar) lo hace con
destilación de la atención antes del ajuste; aquí se parte directamente al preentrenamiento
continuado. Lo que se puede afirmar con este código: la rejilla es exacta como recurrencia, no
añade parámetros, y su inferencia no crece con el contexto. Lo que no: que iguale la perplejidad
de `phi4mini`. Correr los dos presets en paralelo, un workspace cada uno, es la comparación.

## Lo que no está probado

Nada de esto ha corrido en Modal ni ha visto los pesos reales: desde esta sesión no se llega a
Hugging Face. Lo probado en local es la equivalencia matemática del porte (referencia propia) y que
el resto del código no cambia de comportamiento. El primer `modal run` del porte es la prueba real,
y su verificación contra transformers es la que manda.

Riesgos conocidos, por orden:

1. `config.json` de Phi-4-mini con algún campo distinto de lo asumido (`rope_scaling`, sesgos,
   `tie_word_embeddings`): el script aborta con un assert que dice cuál.
2. El trasplante de embedding es una inicialización, no una garantía: el primer tramo puede
   necesitar un LR más bajo para la embedding (hoy `lr_adam = lr_muon/20`).
3. LongRoPE: la cadena entrena a T=1024, dentro del rango "corto" (≤ 4096) donde valen los factores
   cortos; para T > 4096 harían falta los factores largos, que no están implementados.
