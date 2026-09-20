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
