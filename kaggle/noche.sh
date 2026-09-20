#!/bin/bash
# Cadena nocturna sin supervisión: corpus → preentrenamiento continuado → release → chat v4 → charla.
# Cada paso comprueba lo que produjo el anterior y el gasto acumulado antes de encender una GPU.
set -u
cd /Users/alvarez/Desktop/NAVROS-AI
M=".venv/bin/modal"
LOG="results/modal/noche.log"
TOPE=26.0                       # $ del crédito mensual: por encima de esto, no se lanza nada más

log() { echo "[$(date '+%H:%M')] $*" | tee -a "$LOG"; }
gasto() { $M billing summary 2>/dev/null | awk '/Metered/{print $3+0}'; }
estado() { $M app list --json 2>/dev/null | python3 -c "import json,sys; print(next((a['state'] for a in json.load(sys.stdin) if a['app_id']=='$1'),'?'))" 2>/dev/null; }
esperar() { while true; do case "$(estado "$1")" in *stopped*) return 0;; \?) return 1;; esac; sleep 120; done; }
ultima_app() { $M app list --json 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)[0]['app_id'])"; }
hay() { $M volume ls "$1" "$2" 2>/dev/null | grep -q "$3"; }
permitido() { g=$(gasto); log "gasto acumulado: $g \$"; python3 -c "import sys; sys.exit(0 if $g < $TOPE else 1)"; }

log "== cadena nocturna =="

# 1. corpus nuevo (ya en marcha)
if [ -n "${APP_DATOS:-}" ]; then
  log "esperando el corpus ($APP_DATOS)"
  esperar "$APP_DATOS"
fi
if ! hay navros-data corpus_p1 manifest.json; then
  log "ABORTO: el corpus no dejó manifest.json"; exit 1
fi
log "corpus listo: $(grep -oE '\{"es".*' results/modal/pre_datos.log | tail -1)"

# 2. preentrenamiento continuado (4 h ≈ 18 $)
if permitido; then
  log "lanzando preentrenamiento continuado"
  $M run --detach cloud/modal_pre.py::entrenar --horas 4 --total-tokens 800000000 > results/modal/pre_entrenar.log 2>&1 &
  sleep 90; APP_PRE=$(ultima_app); log "app $APP_PRE"; esperar "$APP_PRE"
else
  log "ABORTO: gasto por encima del tope"; exit 1
fi
if ! hay navros-data navros-1b-mas-tokens/export pesos_bf16.pt; then
  log "ABORTO: el preentrenamiento no dejó pesos"; exit 1
fi
log "preentrenamiento hecho: $(grep -oE 'EXPORT .*' results/modal/pre_entrenar.log | tail -1 | cut -c1-300)"

# 3. publicar los pesos nuevos como release
S=/private/tmp/claude-501/-Users-alvarez-Desktop-NAVROS-AI/f9929dd9-f6a0-47ea-a23f-4a8a3fb4a8e2/scratchpad
rm -f $S/pesos_mas_tokens.pt
$M volume get --force navros-data navros-1b-mas-tokens/export/pesos_bf16.pt $S/pesos_mas_tokens.pt >/dev/null 2>&1
$M volume get --force navros-data navros-1b-mas-tokens/export/pesos_bf16.json $S/pesos_mas_tokens.json >/dev/null 2>&1
PASO=$(python3 -c "import json;print(json.load(open('$S/pesos_mas_tokens.json'))['step'])" 2>/dev/null || echo 0)
if [ -f "$S/pesos_mas_tokens.pt" ] && [ "$PASO" != "0" ]; then
  python3 - "$S/pesos_mas_tokens.json" > $S/notas_mas.md <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
v, t = d.get("val") or {}, d.get("test") or {}
print(f"""NAVROS-1B tras preentrenamiento continuado: {d['tokens_vistos']/1e6:.0f}M tokens nuevos sobre los 950M del modelo base (paso {d['step']}), en bf16.

| conjunto | español | inglés |
|---|---|---|
| validación | {v.get('es', float('nan')):.3f} | {v.get('en', float('nan')):.3f} |
| prueba | {t.get('es', float('nan')):.3f} | {t.get('en', float('nan')):.3f} |

- Texto nuevo: partición 1/4 del stream (documentos que el modelo base no vio).
- Corrida nueva desde los pesos publicados, con su propio calentamiento y decaimiento (LR de pico a la mitad).
- SHA-256: `{d['sha256']}`
- Uso: `python scripts/08_conversar.py --pesos pesos_bf16.pt`""")
PY
  gh release create "pesos-1b-mas-tokens-$PASO" "$S/pesos_mas_tokens.pt" --target "$(git rev-parse HEAD)" \
     --title "NAVROS-1B con más tokens (paso $PASO)" --notes-file $S/notas_mas.md >> "$LOG" 2>&1 \
     && log "release publicada: pesos-1b-mas-tokens-$PASO"
fi

# 4. datos de conversación v4 (identidad variada) y ajuste sobre la base nueva
if permitido; then
  log "construyendo datos de chat v4"
  $M run --detach cloud/modal_sft.py::datos --salida /sft/datos_v4 > results/modal/sft_datos4.log 2>&1 &
  sleep 90; APP_D4=$(ultima_app); esperar "$APP_D4"
fi
if hay navros-sft datos_v4 manifest.json && permitido; then
  log "ajustando el chat v4 sobre el modelo con más tokens"
  $M run --detach cloud/modal_sft.py::entrenar --epocas 2 --horas 0.7 --peso-es 0.55 \
     --tag navros-1b-chat-v4 --datos-dir /sft/datos_v4 \
     --base-path /data/navros-1b-mas-tokens/export/pesos_bf16.pt > results/modal/sft_entrenar4.log 2>&1 &
  sleep 90; APP_S4=$(ultima_app); esperar "$APP_S4"
  log "chat v4: $(grep -oE 'EXPORT .*' results/modal/sft_entrenar4.log | tail -1 | cut -c1-300)"
fi

# 5. conversaciones de prueba
if hay navros-sft navros-1b-chat-v4/export pesos_bf16.pt && permitido; then
  $M run cloud/modal_sft.py::charlar --tag navros-1b-chat-v4 > results/modal/sft_charlar4.log 2>&1
  log "charlas guardadas en results/modal/sft_charlar4.log"
fi

log "== fin · gasto final: $(gasto) \$ =="
