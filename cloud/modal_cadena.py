"""Cadena de tramos de NAVROS-1B repartidos entre workspaces de Modal (30 $ cada uno).

Un solo entrenamiento largo, con calentamiento una vez y decaimiento al final, cortado en tramos de
~5,3 h de H100. Cada workspace ejecuta UN tramo; el estado (checkpoint con optimizador, 8 GiB)
viaja entre workspaces por un bucket de GCS, que es lo único que comparten.

    # --- una vez, en el workspace que ya tiene el corpus (o donde se prepare) ---
    modal run cloud/modal_cadena.py::subir_corpus --origen /data/navros_data --nombre p0
    modal run --detach cloud/modal_cadena.py::datos --part 4 --n-parts 12          # corpus nuevo → GCS
    # --- en cada workspace nuevo, tras crear el secreto gcs-navros ---
    modal run cloud/modal_cadena.py::estado
    modal run --detach cloud/modal_cadena.py::tramo
    # --- para vigilar ---
    modal run cloud/modal_cadena.py::estado

Diseño:
  - WSD: LR constante hasta el 90 % de TOTAL_TOKENS y decaimiento lineal en el último 10 %. Los
    tramos intermedios no calientan ni decaen. TOTAL_TOKENS se fija al empezar y NO se cambia
    después de que arranque el decaimiento (subirlo antes es seguro, según navros/lm.py).
  - El corpus es el MISMO conjunto de directorios, en el MISMO orden, en todos los tramos: la
    permutación de los datos depende de ese conjunto (navros/lm.py::TokenData), y cambiarlo a mitad
    rompería la continuidad del cursor. La lista se congela en GCS en el primer tramo y se comprueba
    en los siguientes. Si el corpus se agota, el cursor da la vuelta (repite datos) sin intervención.
  - Dentro del workspace, el checkpoint periódico va al Volume (rápido, y sirve para reanudar si
    Modal reinicia el contenedor: `retries`). Al terminar el tramo se sube a GCS, y solo entonces.
  - El primer tramo parte de los pesos publicados (`--base-url`, `--base-sha`) o de un export del bucket
    (`--base-gcs base/<archivo>.pt`, p. ej. Phi-4-mini portado por cloud/modal_porte_phi.py) con
    optimizador a cero. El preset y la etiqueta se eligen con --preset/--tag o con las variables
    NAVROS_PRESET / NAVROS_TAG (también para `estado`). Una cadena = un preset + una etiqueta.

Requisitos por workspace (los hace la persona, no este código): el secreto `gcs-navros` con la
clave `SERVICE_ACCOUNT_JSON` (misma cuenta de servicio y bucket que cloud/modal_respaldo.py).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/root/navros-ai"
BUCKET = os.environ.get("NAVROS_BUCKET", "navros-respaldo-saasvareoz")
CADENA = "/gcs/navros-cadena"          # raíz de la cadena dentro del bucket
TAG = os.environ.get("NAVROS_TAG", "navros-1b-cadena")   # etiqueta común: sin ella el entrenador no reanuda
PRESET = os.environ.get("NAVROS_PRESET", "navros-1b-fix")  # "phi4mini" para el cuerpo de Phi-4-mini portado
TOTAL_TOKENS = 52_000_000_000          # 52.000M: ~55 tramos de 950M
DECAY_FRAC = 0.10                      # decaimiento en los últimos ~5 tramos
LR_MUON = 0.01                         # como el preentrenamiento continuado (cloud/modal_pre.py)
HORAS = 5.0                            # ≈ 26 $ por tramo (H100 + 4 CPU + 32 GiB); subir tras ver el coste real del primero

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl")
    .pip_install("torch==2.10.0", "numpy", "tokenizers", "datasets")
    .add_local_dir(ROOT / "navros", f"{REMOTE}/navros", ignore=["**/__pycache__/**"])
)
app = modal.App("navros-cadena", image=image)
ckpt_vol = modal.Volume.from_name("navros-cadena-ckpt", create_if_missing=True)
gcs = modal.CloudBucketMount(BUCKET, secret=modal.Secret.from_name("gcs-navros"))


# ----------------------------------------------------------------------------- utilidades
def _sha(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while c := f.read(block):
            h.update(c)
    return h.hexdigest()


def _copia(src: Path, dst: Path, etiqueta: str):
    """Copia atómica (a .tmp y renombra) con tiempo y tamaño en el registro."""
    t0 = time.time()
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)
    print(f"  {etiqueta} {src.stat().st_size / 2**30:.2f} GiB en {time.time() - t0:.0f}s", flush=True)


def _copia_dir(src: Path, dst: Path, etiqueta: str):
    t0 = time.time()
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    tam = sum(p.stat().st_size for p in dst.rglob("*") if p.is_file())
    print(f"  {etiqueta} {tam / 2**30:.2f} GiB en {time.time() - t0:.0f}s", flush=True)


def _lista_corpus() -> list[str]:
    raiz = Path(CADENA) / "corpus"
    return sorted(d.name for d in raiz.iterdir() if (d / "manifest.json").exists()) if raiz.exists() else []


def _registro(evento: dict):
    p = Path(CADENA) / "tramos.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(dict(evento, tiempo=int(time.time()), workspace=os.environ.get("MODAL_WORKSPACE", "?"))) + "\n")


def _setup():
    sys.path.insert(0, REMOTE)
    import torch
    print(dict(torch=torch.__version__, cuda=torch.cuda.is_available(),
               gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None), flush=True)


# ----------------------------------------------------------------------------- corpus
@app.function(cpu=2, memory=8192, timeout=6 * 3600, volumes={"/gcs": gcs, "/data": modal.Volume.from_name("navros-data", create_if_missing=True)})
def subir_corpus(origen: str = "/data/navros_data", nombre: str = "p0"):
    """Copia un corpus ya preparado (manifest.json + shards) del Volume del workspace a GCS."""
    src, dst = Path(origen), Path(CADENA) / "corpus" / nombre
    assert (src / "manifest.json").exists(), f"{origen} no tiene manifest.json"
    assert not (dst / "manifest.json").exists(), f"{nombre} ya está en GCS; usa otro nombre"
    print(f"subiendo {origen} → {dst}", flush=True)
    _copia_dir(src, dst, nombre)
    m = json.loads((dst / "manifest.json").read_text())
    tokens = {l: v["train"]["tokens"] for l, v in m["langs"].items()}
    _registro(dict(evento="corpus", nombre=nombre, tokens=tokens))
    return tokens


@app.function(cpu=8, memory=16384, timeout=12 * 3600, volumes={"/gcs": gcs})
def datos(part: int, n_parts: int = 12, tokens_per_lang: int = 1_500_000_000, horas: float = 5.0):
    """Prepara una partición nueva del stream (como cloud/modal_pre.py::datos) directamente hacia GCS.
    Cada partición es texto distinto; usa números de `part` que no se hayan usado antes."""
    _setup()
    from navros.dataprep import prepare
    nombre = f"p{part}"
    dst = Path(CADENA) / "corpus" / nombre
    assert not (dst / "manifest.json").exists(), f"{nombre} ya existe en GCS"
    tok = Path(CADENA) / "tokenizer.json"
    if not tok.exists():
        shutil.copy(f"{REMOTE}/navros/assets/tokenizer_navros_32k.json", tok)
    out = f"/tmp/{nombre}"
    m = prepare(out, target_tokens_per_lang=tokens_per_lang, time_budget_s=horas * 3600, part=(part, n_parts),
                tokenizer_path=str(tok), log=lambda s: print(s, flush=True))
    _copia_dir(Path(out), dst, nombre)
    tokens = {l: v["train"]["tokens"] for l, v in m["langs"].items()}
    _registro(dict(evento="corpus", nombre=nombre, tokens=tokens))
    print("CORPUS", nombre, tokens, flush=True)
    return tokens


# ----------------------------------------------------------------------------- estado
@app.function(cpu=1, memory=2048, timeout=900, volumes={"/gcs": gcs})
def estado():
    """Qué hay en la cadena: corpus, último checkpoint, tramos hechos y cuántos quedan."""
    corpus = _lista_corpus()
    tokens_corpus = 0
    for c in corpus:
        m = json.loads((Path(CADENA) / "corpus" / c / "manifest.json").read_text())
        tokens_corpus += sum(v["train"]["tokens"] for v in m["langs"].values())
    ck = Path(CADENA) / "ckpt" / TAG / "latest.json"
    meta = json.loads(ck.read_text()) if ck.exists() else None
    tramos = [json.loads(l) for l in (Path(CADENA) / "tramos.jsonl").read_text().splitlines()] if (Path(CADENA) / "tramos.jsonl").exists() else []
    hechos = [t for t in tramos if t.get("evento") == "tramo"]
    tokens_por_paso = 256 * 1024
    vistos = meta["step"] * tokens_por_paso if meta else 0
    resumen = dict(corpus=corpus, tokens_corpus=tokens_corpus, paso=meta["step"] if meta else 0, tokens_vistos=vistos,
                   pasadas_sobre_el_corpus=round(vistos / max(tokens_corpus, 1), 2), tramos_hechos=len(hechos),
                   tramos_que_quedan=round((TOTAL_TOKENS - vistos) / 950e6, 1),
                   decaimiento_empieza_en_tokens=int(TOTAL_TOKENS * (1 - DECAY_FRAC)),
                   ultimo=hechos[-1] if hechos else None)
    print("ESTADO", json.dumps(resumen, indent=1), flush=True)
    return resumen


# ----------------------------------------------------------------------------- tramo
@app.function(gpu="H100", cpu=4, memory=32768, timeout=int((HORAS + 1.5) * 3600),
              volumes={"/ckpt": ckpt_vol, "/gcs": gcs}, retries=modal.Retries(max_retries=3, initial_delay=10.0))
def tramo(horas: float = HORAS, base_url: str = "", base_sha: str = "", base_gcs: str = "",
          total_tokens: int = TOTAL_TOKENS, micro: int = 8, preset: str = PRESET, tag: str = TAG,
          grad_ckpt: bool = False):
    """Un tramo: baja corpus y checkpoint, entrena `horas`, sube el checkpoint y los pesos a GCS.
    Primer tramo: --base-url/--base-sha (release de GitHub) o --base-gcs base/<archivo>.pt (export en el bucket,
    p. ej. el de cloud/modal_porte_phi.py). Para Phi-4-mini portado: --preset phi4mini --micro 4."""
    _setup()
    import torch
    from navros.lm import LMRun
    from navros.lm_ddp import train_ddp
    raiz = Path(CADENA)
    TAG_, PRESET_ = tag, preset
    ck_gcs, ck_local = raiz / "ckpt" / TAG_, Path(f"/ckpt/{TAG_}")
    # el reloj del tramo vive en el Volume: si Modal reinicia el contenedor, el reintento NO vuelve a
    # empezar de cero las `horas` (eso duplicaría el gasto), sino que sigue con lo que quede
    reloj = ck_local / "inicio.json"
    if reloj.exists():
        t_ini = json.loads(reloj.read_text())["t_ini"]
        print(f"reintento: el tramo empezó hace {(time.time() - t_ini) / 3600:.2f} h", flush=True)
    else:
        t_ini = time.time()
        ck_local.mkdir(parents=True, exist_ok=True)
        reloj.write_text(json.dumps(dict(t_ini=t_ini)))
        ckpt_vol.commit()

    # 1) corpus: la misma lista en todos los tramos, copiada a disco local
    lista = _lista_corpus()
    assert lista, "no hay corpus en GCS: ejecuta subir_corpus o datos primero"
    assert lista[0] == "p0", "el primer corpus debe llamarse p0: es el único con val/test (prepare solo los crea en la partición 0)"
    congelada = raiz / "corpus_lista.json"
    if congelada.exists():
        assert json.loads(congelada.read_text()) == lista, (
            f"el corpus cambió desde el primer tramo: {json.loads(congelada.read_text())} → {lista}. "
            "No se puede añadir ni quitar corpus a mitad de cadena.")
    else:
        congelada.write_text(json.dumps(lista))
    dirs = []
    for nombre in lista:
        dst = Path("/tmp/corpus") / nombre
        if not (dst / "manifest.json").exists():
            _copia_dir(raiz / "corpus" / nombre, dst, f"corpus {nombre}")
        dirs.append(str(dst))

    # 2) checkpoint: el del Volume (reintento dentro del workspace) manda; si no, el de GCS; si no, pesos base
    init_from = ""
    if (ck_local / "latest.json").exists():
        print("reanudando desde el Volume de este workspace (reintento)", flush=True)
    elif (ck_gcs / "latest.json").exists():
        print("bajando el checkpoint de GCS", flush=True)
        for f in ("latest.json", "model.pt", "opt_r0.pt"):
            _copia(ck_gcs / f, ck_local / f, f"↓ {f}")
        ckpt_vol.commit()
    elif base_gcs:
        init_from = "/tmp/base_bf16.pt"
        _copia(raiz / base_gcs, Path(init_from), f"↓ base {base_gcs}")
        meta = json.loads((raiz / base_gcs).with_suffix(".json").read_text()) if (raiz / base_gcs).with_suffix(".json").exists() else {}
        if meta.get("sha256"):
            assert _sha(init_from) == meta["sha256"], "la base de GCS no coincide con su SHA-256"
        print(f"primer tramo: base {base_gcs} verificada, optimizador a cero", flush=True)
    else:
        assert base_url and base_sha, "primer tramo: hacen falta --base-url y --base-sha, o --base-gcs"
        init_from = "/tmp/base_bf16.pt"
        subprocess.run(["curl", "-sSL", "--retry", "10", "--retry-all-errors", "-o", init_from, base_url], check=True)
        assert _sha(init_from) == base_sha, "los pesos base no coinciden con el SHA-256 dado"
        print("primer tramo: pesos base verificados, optimizador a cero", flush=True)
    print(f"preparación en {time.time() - t_ini:.0f}s", flush=True)

    # 3) entrenar: el tiempo útil descuenta lo que ya se ha gastado en copias
    restante = horas * 3600 - (time.time() - t_ini)
    assert restante > 600 or (ck_local / "latest.json").exists(), "queda menos de 10 min y no hay nada que subir"
    rc = LMRun(preset=PRESET_, data_dirs=dirs, init_from=init_from, T=1024, batch=256, micro=micro,
               tokens=total_tokens, lr_muon=LR_MUON, lr_adam=LR_MUON / 20, warmup=200, decay_frac=DECAY_FRAC,
               precision="bf16", device="cuda", muon_buf="fp32", grad_ckpt=grad_ckpt,
               eval_every=500, eval_seq=64, r_eval=(1,), log_every=20,
               ckpt_dir=str(ck_local), ckpt_every_min=30, time_budget_s=restante, tag=TAG_)
    res = train_ddp(rc, ns_dtype="bf16", log=lambda s: print(s, flush=True), on_save=ckpt_vol.commit)

    # 4) subir checkpoint (con optimizador) y pesos bf16 a GCS
    for f in ("model.pt", "opt_r0.pt", "latest.json"):   # latest.json el último: es el que marca "válido"
        _copia(ck_local / f, ck_gcs / f, f"↑ {f}")
    so = torch.load(ck_local / "opt_r0.pt", map_location="cpu", mmap=True, weights_only=False)
    sd = {n: t.to(torch.bfloat16) for n, t in so["masters"].items()}
    exp = raiz / "export" / f"paso_{res['step']:06d}"
    exp.mkdir(parents=True, exist_ok=True)
    torch.save(dict(step=res["step"], preset=PRESET_, tag=TAG_, vocab=32768, state=sd), exp / "pesos_bf16.tmp")
    os.replace(exp / "pesos_bf16.tmp", exp / "pesos_bf16.pt")
    info = dict(evento="tramo", paso=res["step"], terminado=res["finished"], tokens_vistos=res["step"] * 256 * 1024,
                val=res.get("val"), test=res.get("test"), horas=round((time.time() - t_ini) / 3600, 2),
                sha256_pesos=_sha(exp / "pesos_bf16.pt"), export=str(exp.relative_to(raiz)))
    (exp / "pesos_bf16.json").write_text(json.dumps(info))
    _registro(info)
    # el Volume de este workspace ya no hace falta: se borra para no pagar almacenamiento
    shutil.rmtree(ck_local, ignore_errors=True)
    ckpt_vol.commit()
    print("TRAMO", json.dumps(info), flush=True)
    return info


@app.local_entrypoint()
def main():
    print(estado.remote())
