# Guion: 2.000 $ en tramos de 30 $ para el 1B (cadena entre workspaces de Modal)

Objetivo: llevar NAVROS-1B de 2.600M a ~52.000M tokens (pérdida proyectada 3,04 → ~2,50; perplejidad
21 → ~12) en unos 55 tramos de ~5 h de H100, uno por workspace. Código: `cloud/modal_cadena.py`.

Todo lo que comparten los workspaces es el bucket de GCS (`navros-respaldo-saasvareoz`, carpeta
`navros-cadena/`). Cada workspace nuevo solo necesita el secreto `gcs-navros`.

## 0. Antes de gastar nada (una vez, en tu Mac)

1. `git pull` de la rama y comprueba que `modal run cloud/modal_cadena.py::estado` importa sin errores
   en un workspace cualquiera (fallará al montar GCS si el secreto no existe: es lo esperado).
2. Ten a mano la URL y el SHA-256 de los pesos base `pesos-1b-2600M` (página de releases de
   `navros-ai`). Se pasan solo al primer tramo.
3. Confirma con Modal que los créditos de desarrollador pueden usarse así. Es lo único que puede
   parar la cadena a mitad.

## 1. Corpus (6 workspaces, ~20 $ cada uno; hacerlo ENTERO antes del primer tramo)

El cargador de datos fija la permutación con el conjunto exacto de corpus. Por eso el corpus se
congela en el primer tramo y no se puede ampliar después. Objetivo: ~30.000M tokens únicos
(52.000M vistos = menos de 2 pasadas).

- Workspace que ya tiene `navros-data` (el original): sube lo que hay.
  ```
  modal secret create gcs-navros SERVICE_ACCOUNT_JSON="$(cat clave.json)"
  modal run cloud/modal_cadena.py::subir_corpus --origen /data/navros_data --nombre p0
  ```
  `p0` es obligatorio y debe ser el corpus con val/test (el original). Si en el workspace
  `heraclutes` hay `corpus_p1`, súbelo desde allí con `--origen /data/corpus_p1 --nombre p1`.
- Cinco workspaces más, una partición nueva cada uno (números que no se hayan usado: 4 en adelante
  si p1..p3 ya existen):
  ```
  modal secret create gcs-navros SERVICE_ACCOUNT_JSON="$(cat clave.json)"
  modal run --detach cloud/modal_cadena.py::datos --part 4 --n-parts 12
  ```
  Cada `datos` produce ~3.000M tokens (1.500M por idioma) en unas 5 h de 8 CPU. Comprueba con
  `estado` que `tokens_corpus` va subiendo. Para cuando `tokens_corpus` ≥ 30.000M.

## 2. Primer tramo (workspace 7)

```
modal secret create gcs-navros SERVICE_ACCOUNT_JSON="$(cat clave.json)"
modal run cloud/modal_cadena.py::estado                      # debe listar el corpus y paso 0
modal run --detach cloud/modal_cadena.py::tramo \
    --base-url https://github.com/navroscol/navros-ai/releases/download/pesos-1b-2600M/<archivo>.pt \
    --base-sha <sha256 del archivo>
```

Vigila este primero de principio a fin en el panel de Modal (`modal app logs navros-cadena`):

- que el corpus se copie en menos de 15 min y el entrenamiento arranque a ~49K tok/s;
- que el checkpoint del Volume se confirme cada 30 min;
- que al final aparezca `TRAMO {...}` con `paso`, `tokens_vistos` y `val`;
- **el coste real del workspace** en la página de facturación. Si quedó por debajo de 26 $, sube
  `--horas` en los siguientes (5,5 h ≈ 28 $); si se pasó, bájalo.

`estado` al terminar debe mostrar `tramos_hechos: 1`, `paso ≈ 3.500` y `tokens_vistos ≈ 900M`.

## 3. Tramos 2 a 55 (un workspace cada uno, 10 minutos de tu tiempo)

Exactamente esto, sin `--base-url`:

```
modal secret create gcs-navros SERVICE_ACCOUNT_JSON="$(cat clave.json)"
modal run cloud/modal_cadena.py::estado
modal run --detach cloud/modal_cadena.py::tramo
```

Lo que hace cada uno: baja el checkpoint (8 GiB) y el corpus de GCS, entrena hasta agotar las horas
con el LR constante, sube el checkpoint y unos pesos bf16 (`export/paso_XXXXXX/`) y borra su Volume.
Si Modal reinicia el contenedor a mitad, el reintento continúa desde el Volume y **no** vuelve a
contar las horas desde cero.

Reglas:

- No lances dos tramos a la vez en workspaces distintos: los dos partirían del mismo checkpoint y
  el segundo pisaría al primero. Uno detrás de otro, comprobando `estado` entre medias.
- No cambies `TOTAL_TOKENS` una vez que `tokens_vistos` supere `decaimiento_empieza_en_tokens`
  (46.800M): el decaimiento ya habrá empezado. Subirlo antes es seguro.
- No toques el corpus. Si un `datos` falla a medias, bórralo del bucket antes del primer tramo.

## 4. Los últimos ~5 tramos: decaimiento

No hay que hacer nada distinto: cuando `tokens_vistos` pasa de 46.800M el LR decae solo hasta 0 en
52.000M. El tramo que llega a `terminado: true` es el último. Ese export es el modelo base final.

## 5. Después (2 workspaces de reserva)

- `modal run cloud/modal_respaldo.py::listar --destino /gcs/navros-cadena` para ver el tamaño; los
  exports intermedios (2 GB cada uno) se pueden borrar salvo el último.
- SFT de chat sobre el export final con `cloud/modal_sft.py`, como hasta ahora.
- Publicar los pesos finales como release y actualizar el README con la curva real de los 55
  tramos (`tramos.jsonl` tiene val y test de cada uno).

## Qué puede salir mal y qué hacer

| síntoma | causa probable | qué hacer |
|---|---|---|
| `tramo` aborta con "el corpus cambió" | se añadió o quitó un corpus en GCS | restaurar la lista de `corpus_lista.json` o, si es intencionado, empezar una cadena nueva con otro `TAG` |
| "primer tramo: hacen falta --base-url" en un workspace que no es el primero | el checkpoint no se subió (el tramo anterior murió antes del paso 4) | volver al workspace anterior y relanzar `tramo`: reanuda del Volume y sube |
| el coste del workspace supera 30 $ | `--horas` alto o copias lentas | bajar `--horas` a 4,5 en el siguiente |
| val sube en vez de bajar durante varios tramos | LR 0,01 demasiado alto para esta fase | no hay ajuste en caliente sin romper la continuidad; parar, decidir, y si acaso empezar una cadena nueva desde el último export con `LR_MUON` menor |

## Variante: cadena sobre el cuerpo de Phi-4-mini (MIT) con nuestro tokenizador

Ver `investigacion/porte_phi4mini.md`. En resumen: primero `modal run cloud/modal_porte_phi.py`
(una vez, CPU), y después la misma cadena con `NAVROS_PRESET=phi4mini NAVROS_TAG=navros-phi4mini-cadena`
y, en el primer tramo, `--base-gcs base/phi_4_mini_instruct_navros32k.pt --micro 4 --total-tokens 14000000000`.
Es una cadena distinta de la del 1B: otra etiqueta, otro preset, el mismo corpus.
