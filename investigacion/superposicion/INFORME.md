# Superposición de estados en un solo recurso: banco de pruebas sobre Collatz

Estudio interno, no enlazado desde el README. Todo lo que hay aquí se mide con el código de este
directorio; las tablas salen de `resumen.py` sobre los JSON de `resultados/`.

## 1. El candidato, en una frase

**La "función de onda" que se pedía ya existe en software clásico y se llama estado de un
producto de matrices (MPS): un vector de χ amplitudes complejas que avanza multiplicando una
matriz por símbolo. Con χ = 16 y 2.600 parámetros aprende el paso de Collatz T(n) exacto en
300 pasos de entrenamiento y lo generaliza sin error a números de 32 bits, habiendo visto solo
hasta 12.** Un Transformer de 100K parámetros entrenado igual no lo consigue fuera de la longitud
de entrenamiento. Los detalles y lo que **no** funciona están en §4.

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

PENDIENTE_TABLAS

## 5. Lectura de los resultados

PENDIENTE_LECTURA

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

Nota: desde el entorno remoto de Claude Code no se llega a `api.modal.com` (la pasarela de salida
de la organización responde 403). El lanzador está escrito y probado en sintaxis, pero **no se ha
ejecutado en Modal**; los resultados de este informe son de CPU local.
