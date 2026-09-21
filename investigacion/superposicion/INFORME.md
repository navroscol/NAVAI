# Superposición de estados en un solo recurso: banco de pruebas sobre Collatz

Estudio interno, no enlazado desde el README. Todo lo que hay aquí se mide con el código de este
directorio; las tablas salen de `resumen.py` sobre los JSON de `resultados/`.

## 1. El candidato, en una frase

**La "función de onda" que se pedía ya existe en software clásico y se llama estado de un
producto de matrices (MPS): un vector de χ amplitudes complejas que avanza multiplicando una
matriz por símbolo. Con χ = 16 y 2.600 parámetros aprende el paso de Collatz T(n) exacto en
300 pasos de entrenamiento y lo generaliza sin error a números de 128 bits, habiendo visto solo
hasta 12.** Un Transformer de 100K parámetros entrenado igual cae al 44 % con 4 bits más y a 0 %
con 12 más. Dos de las ideas de partida **no** sobreviven a la prueba: la versión puramente
"de Fourier" (fases sobre una base común) no computa nada, y la versión "unitaria" (norma
conservada, como una función de onda) no generaliza. Los detalles están en §4 y §5.

## 2. Traducción de las ideas de partida a matemáticas computables

| idea de partida | forma clásica que se probó aquí | archivo |
|---|---|---|
| Un token "vibra" en varios estados a la vez (superposición) | El estado de la secuencia es ψ ∈ ℂ^χ, no un vector de activaciones. Cada símbolo x aplica A[x] ∈ ℂ^{χ×χ}: ψ ← ψ·A[x]. Las χ componentes son hipótesis internas que coexisten y se pesan a la vez. | `modelos.py::CapaMPS` |
| Tensores cuánticos simulados en vez de atención | La secuencia entera es una red de tensores (MPS) contraída por la izquierda y por la derecha; no hay matriz de atención T×T. Variante `unitario`: A[x] = Cayley(H[x]) es unitaria, así que ψ conserva la norma como una función de onda. | `CapaMPS(variante="unitario")` |
| Interferencia constructiva/destructiva antes de emitir | Lectura por la regla de Born: p(bit = c) ∝ \|L_t · M[c] · R_{t+1}\|². Las amplitudes de las hipótesis se suman **antes** de elevar al cuadrado, así que pueden cancelarse. | `modelos.py::born` |
| Series de Fourier: armónicos que coexisten en un punto | Variante `fourier`: todas las A[x] comparten la base propia V y solo cambian las fases: A[x] = V·diag(e^{iθ[x]})·V†. El estado tras t símbolos es literalmente Σ_k c_k·e^{i Σ_s θ_k[x_s]}: una serie de Fourier en las cuentas de símbolos. | `CapaMPS(variante="fourier")` |
| Redes de espigas: no todo el modelo piensa a la vez | Compuerta integra-y-dispara con fuga entre dos capas MPS; solo los canales cuya energía cruza el umbral pasan. Gradiente sustituto (sigmoide) para entrenar. Se mide la tasa de disparo. | `CompuertaEspigas` |
| Criticar la respuesta del otro antes de dar la propia | Dos cabezas MPS. Cada una ve las amplitudes de la otra (módulo y fase) y corrige las suyas; la salida final es la interferencia de ambas. Control: la misma pareja sin crítica (suma directa). | `ModeloDual` |
| Superposición de conceptos (varios conceptos en el mismo recurso) | Modelo de juguete con tres geometrías del estado a igual número de reales: vector, complejo por módulo, matriz de productos exteriores. Resultados en §9. | `conceptos.py` |
| JEPA (predicción en el espacio de representaciones) | **No probado.** Encaja de forma natural: el objetivo sería predecir ψ_{t+k} a partir de ψ_t en el espacio de χ amplitudes, no el token. Queda como siguiente paso. | — |

Lo que hace útil esta traducción es que **el estado ψ de un MPS es exactamente la superposición
de estados de un autómata finito ponderado**: si un problema se resuelve con un autómata de q
estados, existe un MPS con χ ≥ q que lo calcula sin error para cualquier longitud. Eso da una
predicción falsable antes de entrenar nada.

## 3. Por qué Collatz es el banco de pruebas correcto

El paso T(n) = n/2 (par) o 3n+1 (impar), leyendo n en bits de menos a más significativo, es un
**transductor de estado finito**: el bit de paridad decide la rama; n/2 es un desplazamiento;
3n+1 es una suma con un acarreo que vale 0, 1 o 2. Componer k pasos sigue siendo de estado
finito (el autómata producto), pero el número de estados crece con k. Por eso las tareas son:

- `paso-k` (k = 1, 2, 3): dado n en L bits, escribir T^k(n) en L + 2k bits. Etiquetado por
  posición. Predicción: un MPS con χ suficiente lo resuelve **exactamente y para toda longitud**;
  cuanto mayor k, más χ hace falta.
- `parada`: dado n, decidir si su tiempo total de parada supera la mediana de su longitud. No es
  de estado finito (requiere iterar sin cota). Predicción: **ningún** modelo generaliza; en
  distribución solo se puede memorizar.

Protocolo (el mismo del README principal): se entrena con longitudes mezcladas 3..12, se mira
solo la longitud de entrenamiento durante el desarrollo, y las longitudes 16, 24 y 32 se miden
**una vez** al final. 3 semillas, media ± desviación típica. Exactitud = secuencia completa
correcta (todos los bits).

Predicciones concretas, escritas antes de correr el plan:

1. `mps-complejo` y `mps-unitario` alcanzan ~100 % en `paso-1` y lo mantienen a 32 bits.
2. `mps-fourier` **falla** en `paso-1`: sus matrices conmutan, y un sistema conmutativo no puede
   llevar un acarreo (el acarreo depende del orden de los bits). Es la prueba de que "superponer
   armónicos" no basta: hace falta mezcla **no conmutativa** (los tensores), no solo fases.
3. `mps-real` (sin fase, lectura lineal) queda por debajo de las variantes complejas con el mismo χ.
4. Los Transformers (`tf-rope`, `tf-abaco`) aprenden en distribución y se degradan con la longitud.
5. En `parada` nadie supera el azar fuera de distribución.

## 4. Resultados

Plan `base` + `extras` + `extras2`: 84 corridas, CPU local, 3000 pasos, lote 128, χ=16 salvo donde se indica.

### paso-1 — exactitud de secuencia completa (%), media ± desv. típica, 3 semillas

| modelo | χ / d | parámetros | L=12 (entreno) | L=16 | L=24 | L=32 | s/corrida |
|---|---|---|---|---|---|---|---|
| mps-complejo | 16 | 2,624 | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 75 |
| mps-fourier | 16 | 3,184 | 0.5 ± 0.1 | 0.1 ± 0.1 | 0.0 ± 0.0 | 0.0 ± 0.0 | 75 |
| mps-real | 16 | 1,312 | 100.0 ± 0.0 | 100.0 ± 0.0 | 99.9 ± 0.2 | 92.8 ± 0.8 | 47 |
| mps-unitario | 16 | 2,624 | 87.2 ± 8.4 | 23.1 ± 9.8 | 0.0 ± 0.0 | 0.0 ± 0.0 | 74 |
| tf-abaco | 16 | 108,610 | 15.4 ± 2.0 | 4.6 ± 2.4 | 0.1 ± 0.1 | 0.0 ± 0.0 | 72 |
| tf-rope | 16 | 100,418 | 99.9 ± 0.1 | 44.2 ± 18.6 | 1.0 ± 0.7 | 0.0 ± 0.0 | 91 |

### paso-2 — exactitud de secuencia completa (%), media ± desv. típica, 3 semillas

| modelo | χ / d | parámetros | L=12 (entreno) | L=16 | L=24 | L=32 | s/corrida |
|---|---|---|---|---|---|---|---|
| mps-complejo | 16 | 2,624 | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 95 |
| mps-complejo | 32 | 10,368 | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 327 |
| mps-dual-critica | 16 | 5,608 | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 189 |
| mps-dual-suma | 16 | 5,248 | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 180 |
| mps-espigas (disparo 10 %) | 16 | 15,504 | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 219 |
| mps-fourier | 16 | 3,184 | 0.2 ± 0.3 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 92 |
| mps-fourier | 32 | 12,512 | 0.3 ± 0.2 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 337 |
| mps-real | 16 | 1,312 | 100.0 ± 0.0 | 100.0 ± 0.0 | 96.0 ± 5.7 | 83.2 ± 23.8 | 56 |
| mps-unitario | 16 | 2,624 | 89.9 ± 2.0 | 13.5 ± 2.6 | 0.1 ± 0.1 | 0.0 ± 0.0 | 94 |
| tf-abaco | 16 | 108,610 | 0.8 ± 1.1 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 80 |
| tf-rope | 16 | 100,418 | 100.0 ± 0.0 | 67.8 ± 15.5 | 0.9 ± 0.6 | 0.0 ± 0.0 | 101 |

### paso-3 — exactitud de secuencia completa (%), media ± desv. típica, 3 semillas

| modelo | χ / d | parámetros | L=12 (entreno) | L=16 | L=24 | L=32 | s/corrida |
|---|---|---|---|---|---|---|---|
| mps-complejo | 16 | 2,624 | 100.0 ± 0.0 | 99.9 ± 0.1 | 97.2 ± 2.3 | 92.6 ± 5.5 | 110 |
| mps-dual-critica | 16 | 5,608 | 99.9 ± 0.1 | 99.3 ± 0.8 | 97.1 ± 2.4 | 89.4 ± 6.0 | 290 |
| mps-dual-suma | 16 | 5,248 | 100.0 ± 0.0 | 99.2 ± 1.0 | 95.6 ± 2.7 | 84.5 ± 5.6 | 249 |
| mps-espigas (disparo 6 %) | 16 | 15,504 | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 253 |
| mps-fourier | 16 | 3,184 | 0.3 ± 0.2 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 111 |
| mps-real | 16 | 1,312 | 100.0 ± 0.0 | 98.2 ± 0.6 | 82.0 ± 6.9 | 59.0 ± 15.2 | 65 |
| mps-unitario | 16 | 2,624 | 48.2 ± 5.2 | 4.6 ± 0.7 | 0.0 ± 0.0 | 0.0 ± 0.0 | 110 |
| tf-abaco | 16 | 108,610 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 89 |
| tf-rope | 16 | 100,418 | 86.3 ± 2.0 | 2.9 ± 0.9 | 0.0 ± 0.0 | 0.0 ± 0.0 | 112 |

### parada — exactitud de secuencia completa (%), media ± desv. típica, 3 semillas

| modelo | χ / d | parámetros | L=12 (entreno) | L=16 | L=24 | L=32 | s/corrida |
|---|---|---|---|---|---|---|---|
| mps-complejo | 16 | 2,688 | 77.6 ± 2.3 | 55.1 ± 4.1 | 49.3 ± 1.0 | 50.9 ± 0.5 | 31 |
| tf-rope | 16 | 100,418 | 64.8 ± 0.9 | 52.6 ± 3.6 | 50.4 ± 1.3 | 49.4 ± 1.4 | 74 |

### paso-2 — exactitud por bit (%), media ± desv. típica, 3 semillas

| modelo | χ / d | parámetros | L=12 (entreno) | L=16 | L=24 | L=32 | s/corrida |
|---|---|---|---|---|---|---|---|
| mps-complejo | 16 | 2,624 | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 95 |
| mps-complejo | 32 | 10,368 | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 327 |
| mps-dual-critica | 16 | 5,608 | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 189 |
| mps-dual-suma | 16 | 5,248 | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 180 |
| mps-espigas (disparo 10 %) | 16 | 15,504 | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 219 |
| mps-fourier | 16 | 3,184 | 68.9 ± 0.4 | 63.8 ± 0.5 | 58.1 ± 0.4 | 56.3 ± 0.2 | 92 |
| mps-fourier | 32 | 12,512 | 69.5 ± 0.3 | 63.7 ± 0.7 | 57.2 ± 1.0 | 56.0 ± 1.2 | 337 |
| mps-real | 16 | 1,312 | 100.0 ± 0.0 | 100.0 ± 0.0 | 99.6 ± 0.5 | 97.8 ± 3.1 | 56 |
| mps-unitario | 16 | 2,624 | 99.2 ± 0.2 | 90.0 ± 0.5 | 69.0 ± 0.9 | 59.2 ± 0.6 | 94 |
| tf-abaco | 16 | 108,610 | 69.5 ± 2.0 | 65.7 ± 1.6 | 61.7 ± 0.9 | 59.1 ± 1.0 | 80 |
| tf-rope | 16 | 100,418 | 100.0 ± 0.0 | 96.2 ± 3.1 | 75.1 ± 3.1 | 66.7 ± 0.7 | 101 |


### Longitud extrema — `mps-complejo` χ=16, 1 semilla, entrenado con 3..12 bits

Exactitud de secuencia completa / por bit (%). Los números de 256 bits tienen ~77 cifras decimales.

| tarea | L=12 | L=32 | L=64 | L=128 | L=256 |
|---|---|---|---|---|---|
| paso-1 | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 / 100.0 | 91.4 / 99.6 |
| paso-2 | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 / 100.0 | 74.6 / 96.4 |


## 5. Lectura de los resultados

Predicción por predicción:

1. **Confirmada a medias.** `mps-complejo` da 100 % en `paso-1` y `paso-2` en todas las
   longitudes, y en la prueba de longitud extrema sigue al 100 % hasta **128 bits** (10× lo visto
   en entrenamiento); a 256 bits acierta el 99,6 % de los bits y el 91 % de las secuencias. No es
   exacto en el límite: la normalización paso a paso y el redondeo en coma flotante acumulan
   deriva. **`mps-unitario` falla**, y la razón es instructiva: un transductor determinista
   **fusiona** estados (el acarreo se reinicia, la información se descarta) y una matriz unitaria
   no puede fusionar nada porque es invertible. La "función de onda que conserva la norma" es
   justo la restricción equivocada para computar; hace falta evolución **disipativa** (matrices
   no unitarias). Cuanto más pasos (k), peor le va al unitario (87 → 90 → 48 % en distribución).
2. **Confirmada.** `mps-fourier` queda en ~0 % de secuencias en las tres tareas, con χ=16 y χ=32,
   y en ~69 % de bits, que es lo que se saca de la paridad y poco más. Superponer armónicos con una
   base propia común es un sistema conmutativo, y el acarreo no conmuta con el orden de los bits.
   La superposición útil necesita mezcla no conmutativa, o sea tensores, no solo fases.
3. **Confirmada.** `mps-real` aprende en distribución pero se degrada fuera (92,8 %, 83,2 % y
   59,0 % a 32 bits en k = 1, 2, 3, con mucha varianza entre semillas) donde el complejo se queda
   en 100 %, 100 % y 92,6 %. Misma χ, la mitad de parámetros: la fase compra estabilidad de la
   solución, no solo capacidad.
4. **Confirmada.** `tf-rope` llega al 100 % (k = 1, 2) y 86 % (k = 3) en distribución, y cae a
   44 %, 68 % y 3 % con solo 4 bits más, y a 0 % con 24 bits. `tf-abaco` ni siquiera aprende en
   distribución con esta configuración (2 capas, d = 64, 3000 pasos): es una referencia débil, no
   ajustada, y no debe leerse como un resultado sobre el ábaco del README principal.
5. **Confirmada.** En `parada`, el MPS memoriza en distribución (77,6 %) y el Transformer menos
   (64,8 %); a 16 bits ya están en 55 % y 53 %, y a 24 y 32 bits en el azar exacto (49 a 51 %).

Lo que no estaba previsto y salió de las pruebas extra:

- **La crítica cruzada no aporta nada medible.** En `paso-3` (donde χ=16 no satura), dos cabezas
  con crítica dan 89,4 ± 6,0 % a 32 bits, dos cabezas sumadas sin crítica 84,5 ± 5,6 %, y **una
  sola cabeza 92,6 ± 5,5 %**. Las diferencias están dentro del ruido entre semillas y, si acaso,
  van en contra: duplicar cabezas duplica el número de soluciones que hay que hacer coincidir. Con
  esta forma de crítica (una corrección lineal de amplitudes a partir de la propuesta de la otra)
  la idea no se sostiene; si se quiere seguir, hay que cambiar el mecanismo, no el tamaño.
- **Las espigas sí funcionan, pero por la profundidad.** Dos capas MPS con compuerta LIF entre
  ambas dan 100 % en `paso-3` a 32 bits (tres semillas, sin varianza) con solo el **6 % de los
  canales disparando**; la capa única complejo da 92,6 %. Es el único modelo que resuelve `paso-3`
  del todo. Cuidado con la lectura: tiene 6× más parámetros y dos capas; lo que demuestra es que
  una red apilada de superposiciones **puede** funcionar con el 94 % de sus canales en silencio,
  no que el silencio sea lo que la hace mejor. Falta el control de dos capas sin compuerta.
- **χ=32 no cambia nada en `paso-2`** (ya saturado) y cuesta 3,4× más tiempo por corrida.

Lo que este estudio **no** demuestra:

- No hay comparación a igual cómputo ni a igual número de parámetros: el MPS gana con 40× menos
  parámetros, pero el Transformer no se ajustó (un solo LR, 2 capas). El resultado robusto es la
  **forma** de las curvas de longitud, no la cifra en distribución.
- Solo hay una semilla en la prueba de longitud extrema.
- Los modelos son diminutos y la tarea es binaria; nada de esto se ha probado en texto.
- No se ha ejecutado en Modal: el lanzador existe, pero la red del entorno remoto lo bloquea (§8).

## 6. Qué dice esto sobre el Problema 18 de Smale y la "pérdida no computable"

Lo que el banco de pruebas permite afirmar, y solo eso:

- **Hay una dicotomía medible.** Las funciones de estado finito (cada paso de Collatz, y sus
  composiciones) se aprenden **exactamente** por descenso de gradiente con un modelo cuyo estado es
  una superposición de estados de autómata, y la solución encontrada generaliza a longitudes
  nunca vistas. La función global (tiempo de parada) no: ningún modelo sale del azar fuera de la
  distribución de entrenamiento, y en distribución lo que sube es memorización.
- **La frontera no la pone el optimizador sino la clase de funciones.** Con la misma pérdida, el
  mismo optimizador y los mismos datos, el modelo pasa de 100 % a 50 % al cambiar la tarea de
  "un paso" a "cuántos pasos". Eso es coherente con la reducción al problema de la parada: no hay
  un autómata finito (ni una red de tamaño fijo) que compute el tiempo de parada de un mapa de
  Collatz generalizado, porque la generalización de Conway es Turing-completa.
- **Sobre la fórmula ℒ(θ) = lim E[D(T^k(n), objetivo)] = ∞.** Hay que revisarla antes de usarla:
  para el mapa de Collatz estándar nadie ha probado que exista un n para el que T^k(n) diverja
  (la conjetura es que no existe), así que el límite no está demostrado infinito; y una pérdida
  que se define con un límite k→∞ no es la pérdida que minimiza SGD, que siempre entrena con k
  finito. El argumento sólido es otro: **la pérdida sí es computable para cada k finito; lo que no
  es computable es decidir si existe θ que la haga cero para todo k a la vez**, porque eso
  equivale a decidir la parada. El banco de pruebas es una ilustración empírica de esa
  separación, no una prueba.
- Collatz en sí sigue abierto. Lo indecidible (Conway 1972, y la formalización de Kurtz y Simon
  2007) es la familia generalizada de mapas por residuos; el mapa 3n+1 concreto no está probado
  indecidible. Conviene no mezclar ambas cosas al citar.

Propuesta que sale de aquí para que se verifique con fuentes de ingeniería y datos: **medir la
inteligencia de una arquitectura por el mayor k para el que aprende T^k exacto y general**, no
por la exactitud en distribución. Es una escala ordinal (k = 1, 2, 3, ...) ligada al número de
estados del autómata producto, y se puede comparar entre familias (MPS, Transformer, recurrente)
a igual cómputo.

## 9. Superposición de conceptos: ¿caben más en un estado bidimensional?

> Corregido en §10: la fragilidad del vector real a n = 144 y su peor pérdida en el régimen denso eran
> del LR (1e-2), no de la geometría. Se deja el texto original como registro.

Segunda prueba, pedida tras el banco de Collatz: medir la superposición directamente sobre las
representaciones, no sobre tareas. Es el modelo de juguete de Elhage et al. (m rasgos dispersos
comprimidos en n números reales y reconstruidos con ReLU), con tres geometrías del estado a igual
presupuesto de n reales: vector real, vector complejo leído por módulo (bidimensional por fase) y
matriz √n×√n donde cada concepto es un producto exterior (bidimensional por rejilla). Todo el
barrido (3 geometrías × 3 relaciones m/n × 7 dispersiones × 3 semillas) se entrena vectorizado
como un tensor por lotes. Código en `conceptos.py`; cuaderno en `kaggle_conceptos.ipynb`;
tablas completas en `resultados/conceptos_kaggle.md` (Kaggle, 2×T4, n = 64 y n = 144).

Predicciones escritas antes de correrlo: (a) la fase permitiría que las interferencias entre
conceptos se sumaran de forma incoherente y cupieran más conceptos a la vez; (b) la rejilla
tendría interferencia menor porque el solape entre dos conceptos es un producto de dos cosenos.

Lo que salió:

- **La fase no ayuda; con lectura por módulo estorba.** A n = 64 y m/n = 4, S = 0,9, el vector
  complejo representa el 21 % de los rasgos (real: 33 %), tiene una interferencia 5× mayor
  (0,033 contra 0,006) y el error con k conceptos activos a la vez crece más deprisa (0,32 por
  concepto contra 0,22). A n = 144 pasa lo mismo. La razón es de fondo: el módulo tira la
  información de signo, así que la fase no puede usarse para leer un concepto, solo para que los
  demás se cancelen entre sí, y eso no compensa. Predicción (a) **falsada** en esta forma. Cabe
  recordar que la versión "compleja con lectura lineal" es matemáticamente idéntica al vector
  real con n dimensiones, así que no hay una tercera opción que probar sin cambiar la lectura.
- **La rejilla tiene la interferencia más baja, por un orden de magnitud.** Entre rasgos
  representados, la matriz queda en 0,000 a 0,007 (n = 64) y 0,000 a 0,018 (n = 144) en todo el
  rango de dispersión, contra 0,002 a 0,03 del vector real en el régimen disperso y > 0,4 en el
  denso. Predicción (b) **confirmada**. El precio: representa menos conceptos (a n = 144, m/n = 4:
  15 % contra 20 a 100 % del real) y los guarda casi ortogonales (1,4 a 1,9 dimensiones por
  rasgo: no está superponiendo, está seleccionando). Ojo con el conteo de parámetros: un concepto
  en la rejilla cuesta 2√n números (24 a n = 144) frente a n (144) en el vector, 6× menos.
- **En el régimen denso las dos geometrías bidimensionales dan menor pérdida que el vector
  real.** Con S = 0 y m/n = 4 a n = 64: real 0,013, complejo 0,001, matriz 0,000; a n = 144 con
  m/n = 2: real 0,012, complejo 0,003, matriz 0,000. En el régimen disperso (S ≥ 0,9) las tres
  llegan a ~0 y no hay nada que las distinga por pérdida: **ninguna geometría guarda más
  conceptos que otra en el mismo recurso cuando los conceptos son dispersos**, que es el caso
  que importa en interpretabilidad.
- **El vector real a n = 144 encontró soluciones frágiles.** Con S ≤ 0,9 el real "representa" el
  100 % de los rasgos con 0,05 a 0,2 dimensiones por rasgo e interferencias > 1: normas enormes
  que se compensan entre sí y reconstruyen bien en distribución (pérdida 0,001) pero estallan
  cuando hay menos conceptos activos de lo habitual: error de 2.146 con un solo concepto activo,
  contra 0,5 del complejo y 0,4 de la matriz. Las dos geometrías bidimensionales nunca entraron
  en ese régimen. Puede ser un artefacto del optimizador (un solo LR, 10.000 pasos), pero las
  tres semillas coinciden y es la observación más interesante del barrido: **las lecturas
  bidimensionales acotan la solución** (el módulo es no negativo; el bilineal no puede compensar
  normas grandes con signos), y eso da soluciones que sobreviven fuera de distribución.

Defectos del propio experimento, para no sobreleer:

- El error con k conceptos activos elige los k entre **todos** los rasgos, incluidos los de
  importancia ínfima que ningún modelo guarda. Por eso su pendiente coincide casi exactamente con
  la fracción de rasgos no representados (0,67 × 1/3 ≈ 0,22 por concepto en el real). Mide
  cobertura, no interferencia. Hay que repetirlo muestreando entre los rasgos importantes.
- A igual n, las geometrías no tienen igual número de parámetros ni igual familia de funciones.
  Falta la comparación a igual parámetros (rejilla de rango 2 o 3 contra vector) y el vector real
  con un LR más bajo para descartar el artefacto del punto anterior.
- Nada de esto es texto: son rasgos sintéticos con importancia geométrica 0,9^i.

Lo que queda del candidato tras esta prueba: **la rejilla (estado como matriz, conceptos como
productos exteriores) es la única forma "bidimensional" que sale bien parada**, por interferencia
mínima, parámetros por concepto y robustez; no por capacidad. La fase, tal como se probó, no.

## 10. Superposición de conceptos, ronda 2: las tres pruebas pendientes

Mismo cuaderno con tres añadidos (`kaggle_conceptos2.ipynb`, tablas en
`resultados/conceptos2_kaggle.md`): k conceptos activos elegidos solo entre los 32 importantes; rejilla
de rango r con r = √n/2 para igualar los n parámetros por concepto del vector (`matriz-r4` a n = 64,
`matriz-r6` a n = 144); y el vector real con LR 1e-3 (`real-lr3`).

Lo que cambia respecto a §9, y hay que corregir allí:

- **La fragilidad del vector real a n = 144 era del optimizador, no de la geometría.** Con LR 1e-3
  el vector real es el mejor modelo del barrido en todo: pérdida 0,000 en los 21 regímenes,
  interferencia uniforme de 0,04 a 0,06, error **exactamente cero** con hasta 8 conceptos
  importantes activos a la vez, y el menor error entre todos los rasgos (0,331 con k = 1). La
  ventaja de las geometrías bidimensionales en el régimen denso de §9 también desaparece: era
  el mismo artefacto (LR 1e-2 demasiado alto para n ≥ 64). Lo que sí queda de §9 es más modesto:
  las lecturas por módulo y bilineal **toleran un LR 10× peor** sin degenerar. Es robustez al
  ajuste, no capacidad.
- **Entre los conceptos importantes no hay interferencia medible en ninguna geometría real.** El
  error con k importantes activos es plano en k para real, real-lr3 y las tres rejillas, a n = 64 y
  a n = 144: el valor constante (0,01 a 0,2) es el sesgo positivo de los rasgos no representados,
  que emiten una constante pase lo que pase, no interferencia. Solo el complejo por módulo crece
  con k (0,051 → 0,591 a n = 64). Pero ojo: 32 conceptos importantes caben ortogonales en 64 o
  144 dimensiones, así que la prueba no fuerza superposición **entre los importantes**. Para
  hacerlo hay que aplanar la importancia (0,98^i) o declarar importantes 2n rasgos. Sigue
  pendiente.
- **La rejilla a igual número de parámetros se comporta como el vector.** `matriz-r6` a n = 144
  representa el 100 % de los rasgos en el régimen denso, con interferencias de 0,1 a 0,7 y pérdidas
  iguales a las del vector con LR alto. El rango es un mando: rango 1 da la interferencia mínima,
  6× menos parámetros y pocos conceptos; rango √n/2 recupera la capacidad del vector y sus
  problemas. No hay almuerzo gratis; hay una familia intermedia que no existía en el vector.
- **En cobertura (error entre todos los rasgos), la rejilla de rango alto gana por poco:** 0,297
  contra 0,331 con k = 1 a n = 144 y 0,266 contra 0,347 por concepto añadido. El complejo es el
  peor en todas las condiciones.

Conclusión de la línea de conceptos, con los datos de las dos rondas: **ninguna geometría
bidimensional guarda más conceptos dispersos en el mismo recurso que un vector real bien
ajustado**. La fase por módulo perjudica. La rejilla de productos exteriores es una
parametrización legítima que cambia interferencia por capacidad con un mando (el rango) y
resiste mal ajuste, y eso es todo lo que se puede afirmar. La pregunta original ("varios
significados a la vez en el mismo token") no se responde con capacidad estática; el sitio donde
sí apareció algo fue en la dinámica (§4 y §5: el estado MPS como superposición de estados de
autómata), y ahí conviene volver.

## 11. Escala ordinal medida y proyección a 2B-40B

### 11.1 La escala ordinal, medida en CPU (1 semilla)

Mayor k tal que el MPS complejo de anchura χ resuelve T^k exacto en distribución y general a 32
bits. Exactitud de secuencia completa 12 bits / 32 bits (%). Plan `escala-local`; el plan completo
`escala` (χ hasta 64, k hasta 6, 3 semillas, con Transformer de referencia) está escrito para Modal.

| χ | parámetros | T¹ | T² | T³ | T⁴ | T⁵ | mayor k exacto y general (≥ 99 % a 32 bits) |
|---|---|---|---|---|---|---|---|
| 8 | 672 | 100 / 100 | 100 / 100 | 72 / 40 | 71 / 2 | 29 / 2 | **2** |
| 16 | 2,624 | 100 / 100 | 100 / 100 | 100 / 99 | 94 / 81 | 86 / 44 | **3** |
| 32 | 10,368 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 96 / 73 | **4** |

La escala existe y es limpia: cada duplicación de χ compra aproximadamente un paso más de
composición exacta. Es coherente con la teoría: el autómata producto de k pasos de Collatz tiene
del orden de 3^k estados de acarreo, y hace falta χ ≳ número de estados para representarlo; χ = 8
cubre k = 2 (9 estados), χ = 16 no llega a k = 3 del todo (27 estados) pero casi, χ = 32 cubre k = 4
(81 estados) sin error. Esa correspondencia es la propuesta de medida para el Problema 18: **la
"inteligencia" de una arquitectura sobre esta familia es el mayor k que compone exacto y general,
y crece con el logaritmo de su estado interno, no con sus parámetros**. El plan `escala` completo
mide dónde se sitúa el Transformer en la misma escala (la predicción, por §4, es k = 0 fuera de
distribución para todo d).

### 11.2 Proyección a 2B-40B parámetros: tiempo, coste y pérdida

No es una medida: es una extrapolación con dos anclas, lo medido en el README de este proyecto
para el 1B en H100 (49K tokens/s, 320 TFLOP/s útiles, 24,3 $ por 5,5 h) y la ley de escala de
Chinchilla calibrada con la pérdida real del 1B (predice 2,987 nats donde se midieron 3,040; se
aplica ese desplazamiento). Código en `proyeccion.py`; tabla completa en `resultados/proyeccion.md`.

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


### 11.3 ¿Sería más inteligente a 2B-40B con el método bidimensional o de rejilla?

La respuesta honesta, con los datos de este estudio, tiene tres partes.

1. **La proyección de pérdida de arriba es para un Transformer; para la rejilla no hay ley de
   escala propia medida aquí.** Lo único que se puede afirmar es lo que se midió: en superposición
   estática de conceptos (§9 y §10) la rejilla no guarda más que un vector bien ajustado, así que
   no hay razón medida para esperar mejor perplejidad a igual parámetros. Lo que sí tiene es
   interferencia menor y tolerancia a mal ajuste; a gran escala, donde el LR se ajusta bien, esa
   ventaja vale poco.
2. **La propuesta a escala ya tiene parientes publicados y con nombre.** Una capa con estado
   matricial que acumula productos exteriores k·vᵀ y lee con una consulta es exactamente la
   atención lineal y sus descendientes (GLA, DeltaNet, RWKV-6/7, Mamba-2 en su forma de espacio de
   estados); y el MPS con matrices dependientes del símbolo es una recurrencia lineal con
   transición dependiente de la entrada, que es la familia Mamba/LRU. Según esa literatura, a
   verificar con fuentes propias, a igual parámetros (1 a 7B) igualan la perplejidad del
   Transformer con diferencias de pocas centésimas de nat, ganan en rendimiento y memoria con
   contextos largos y en extrapolación de longitud, y pierden en recuperación exacta de
   información del contexto (copiar, buscar una aguja). **Nada en este estudio contradice ese
   cuadro, y el resultado de Collatz lo ilustra en pequeño: gana donde la tarea es un autómata,
   no donde hay que memorizar.**
3. **Por tanto, la apuesta racional no es "40B con rejilla" sino híbrida y barata de comprobar:**
   entrenar dos modelos de 100 a 200M parámetros con el entrenador que ya existe (`navros/lm.py`),
   uno con el bloque actual y otro con el bloque de rejilla en el lugar de la atención, al mismo
   cómputo, y medir perplejidad en distribución y en secuencias 4× más largas que las de
   entrenamiento. Cuesta menos de 20 $ de H100 y responde la pregunta que ninguna proyección
   puede responder. Si la rejilla no pierde perplejidad y gana longitud, entonces el 2B con un
   bloque híbrido (rejilla en la mayoría de capas, atención en unas pocas para la recuperación
   exacta) tiene sentido, y su coste está en la tabla: unos 2.200 $ y 3 días con 8 H100 a tokens
   óptimos.

Sobre los tiempos de la tabla, en llano: el salto del 1B actual a un 2B a tokens óptimos son
40.000M tokens, 15× los que se han usado hasta hoy, y 30× el coste del último tramo. 7B es
26.000 $; 40B, casi 900.000 $ y meses con 8 GPUs. Kaggle no sirve para nada de esto. Y el 2B con
los 2.600M tokens actuales (141 $) quedaría en 2,97 nats, peor que el 1B con tokens óptimos
(541 $, 2,63): **con este presupuesto, el mejor uso del dinero son tokens para el 1B, no
parámetros**.

## 7. Referencias que hay que verificar con fuentes propias

Las cito de memoria; antes de apoyarse en ellas hay que comprobarlas.

- Stoudenmire y Schwab, *Supervised learning with tensor networks*, NeurIPS 2016 (MPS como modelo
  de aprendizaje).
- Bradley, Stoudenmire y Terilla, *Modeling sequences with quantum states*, 2020 (MPS como modelo
  de lenguaje; la conexión con autómatas ponderados).
- Elhage et al., *Toy models of superposition*, Anthropic 2022 (superposición en interpretabilidad).
- Arjovsky, Shah y Bengio, *Unitary evolution recurrent neural networks*, ICML 2016 (recurrencias
  unitarias; la variante `unitario` es un caso sin no linealidad).
- Neftci, Mostafa y Zenke, *Surrogate gradient learning in spiking neural networks*, 2019.
- LeCun, *A path towards autonomous machine intelligence*, 2022 (JEPA).
- Conway, *Unpredictable iterations*, 1972 (FRACTRAN y la generalización indecidible de Collatz).
- Kurtz y Simon, *The undecidability of the generalized Collatz problem*, TAMC 2007.
- Smale, *Mathematical problems for the next century*, 1998 (Problema 18).

## 8. Reproducir

```bash
# local, CPU (unos 30 min con 4 núcleos)
python investigacion/superposicion/experimentos.py --plan base
python investigacion/superposicion/experimentos.py --plan extras
python investigacion/superposicion/resumen.py base extras

# Modal: abanico de contenedores; `grande` (χ=32, L=16→48, 6000 pasos, 90 corridas) en T4
modal run investigacion/superposicion/modal_superposicion.py --plan base
modal run investigacion/superposicion/modal_superposicion.py --plan grande --gpu
```

Nota: desde el entorno remoto del agente no se llega a `api.modal.com` (la pasarela de salida
de la organización responde 403). El lanzador está escrito y probado en sintaxis, pero **no se ha
ejecutado en Modal**; los resultados de este informe son de CPU local.
