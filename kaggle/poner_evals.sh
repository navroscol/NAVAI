#!/bin/bash
# Copia al corpus indicado la validación y la prueba del corpus original y parchea su manifest.
#   bash kaggle/poner_evals.sh <perfil> <volumen> <carpeta>     p. ej. herclues navros-data corpus_p2
# `prepare` solo crea val/test en la partición 0; reutilizar siempre los mismos conjuntos, que
# ningún modelo ha entrenado, mantiene las pérdidas comparables entre tramos.
set -eu
PERFIL="${1:-herclues}"; VOL="${2:-navros-data}"; DIR="${3:-corpus_p2}"
ORIG=/Users/alvarez/Desktop/NAVROS-copia/modal/navros-data/navros_data
S=/private/tmp/claude-501/-Users-alvarez-Desktop-NAVROS-AI/f9929dd9-f6a0-47ea-a23f-4a8a3fb4a8e2/scratchpad
M="/Users/alvarez/Desktop/NAVROS-AI/.venv/bin/modal"
export MODAL_PROFILE="$PERFIL"

for f in es/val.bin es/test.bin en/val.bin en/test.bin; do
  $M volume put "$VOL" "$ORIG/$f" "/$DIR/$f" >/dev/null 2>&1 && echo "subido $f"
done

$M volume get --force "$VOL" "$DIR/manifest.json" "$S/man_$DIR.json" >/dev/null 2>&1
python3 - "$ORIG/manifest.json" "$S/man_$DIR.json" "$DIR" <<'PY'
import json, sys
orig, destino, dirname = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), sys.argv[3]
for lang, v in orig["langs"].items():
    for split in ("val", "test"):
        destino["langs"][lang][split] = v[split]
destino["eval_de"] = "corpus original (partición 0/4)"
json.dump(destino, open(sys.argv[2], "w"), indent=1, ensure_ascii=False)
print({l: dict(train_B=round(x["train"]["tokens"] / 1e9, 3), val=x["val"]["file"]) for l, x in destino["langs"].items()})
PY
$M volume put --force "$VOL" "$S/man_$DIR.json" "/$DIR/manifest.json" >/dev/null 2>&1 && echo "manifest parcheado en $VOL:$DIR"
