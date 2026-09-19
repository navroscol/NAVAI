"""Corpus bilingüe pretokenizado para NAVROS-LM (se ejecuta en un notebook de CPU de Kaggle).

Por idioma, en orden de lectura del stream:
  1. primeros documentos hasta `tok_bytes`  → entrenan el tokenizador (y luego también el LM)
  2. siguientes 2000 documentos              → verificación ida y vuelta del tokenizador
  3. siguientes hasta `eval_tokens`          → val.bin   (nunca se entrena con ellos)
  4. siguientes hasta `eval_tokens`          → test.bin  (nunca se usan para elegir nada)
  5. el resto hasta el objetivo             → train_###.bin
Cada documento termina en <|eot|> (id 0). Tokens en uint16 crudo. Un manifest.json
registra fuentes, recuentos, compresión y el SHA-256 de cada shard.
"""
from __future__ import annotations

import hashlib
import json
import time
from itertools import islice
from pathlib import Path

import numpy as np

SOURCES = {
    "es": [("epfml/FineWeb2-HQ", "spa_Latn"), ("HuggingFaceFW/fineweb-2", "spa_Latn")],
    "en": [("HuggingFaceFW/fineweb-edu", "sample-100BT"), ("HuggingFaceFW/fineweb-edu", "sample-10BT")],
}
SHARD_TOKENS = 1 << 27  # 134M tokens = 256 MiB por shard


def open_stream(lang, part=(0, 1)):
    """part = (i, n): esta ejecución lee solo los archivos i, i+n, i+2n… del dataset, para que
    varios notebooks de datos no repitan documentos."""
    from datasets import load_dataset
    errors = []
    for name, cfg in SOURCES[lang]:
        try:
            ds = load_dataset(name, name=cfg, split="train", streaming=True)
            if part[1] > 1:
                ds = ds.shard(num_shards=part[1], index=part[0])
            it = (ex["text"] for ex in ds)
            first = next(it)
            return _chain([first], it), f"{name}/{cfg}"
        except Exception as e:
            errors.append(f"{name}/{cfg}: {type(e).__name__}: {str(e)[:200]}")
    raise RuntimeError("ninguna fuente disponible para " + lang + ": " + " | ".join(errors))


def _chain(head, tail):
    yield from head
    yield from tail


def take_bytes(it, n_bytes):
    out, total = [], 0
    for t in it:
        out.append(t)
        total += len(t.encode("utf-8"))
        if total >= n_bytes:
            break
    return out


def sha256_file(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


class ShardWriter:
    def __init__(self, folder, prefix, max_tokens=None):
        self.folder, self.prefix, self.max = Path(folder), prefix, max_tokens
        self.folder.mkdir(parents=True, exist_ok=True)
        self.buf, self.n_buf, self.shards, self.total = [], 0, [], 0

    def add(self, ids_list):
        for ids in ids_list:
            self.buf.append(np.asarray(ids + [0], dtype=np.uint16))
            self.n_buf += len(ids) + 1
            self.total += len(ids) + 1
            if self.max is None and self.n_buf >= SHARD_TOKENS:
                self.flush()

    def full(self):
        return self.max is not None and self.total >= self.max

    def flush(self):
        if not self.buf:
            return
        arr = np.concatenate(self.buf)
        name = f"{self.prefix}.bin" if self.max is not None else f"{self.prefix}_{len(self.shards):03d}.bin"
        path = self.folder / name
        arr.tofile(path)
        self.shards.append(dict(file=f"{self.folder.name}/{name}", tokens=int(arr.size), sha256=sha256_file(path)))
        self.buf, self.n_buf = [], 0


def prepare(out_dir, target_tokens_per_lang=2_000_000_000, tok_bytes=150_000_000, eval_tokens=5_000_000,
            vocab_size=32768, time_budget_s=10.5 * 3600, batch_docs=2000, part=(0, 1), tokenizer_path=None, log=print):
    """part=(0, n) entrena el tokenizador y crea val/test; part=(i>0, n) reutiliza tokenizer_path
    y solo produce shards de entrenamiento de su partición."""
    from tokenizers import Tokenizer
    from .tokenizer import train_tokenizer, verify_roundtrip
    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    first = part[0] == 0
    streams, heads, manifest = {}, {}, dict(sources={}, langs={}, part=list(part))
    for lang in SOURCES:
        streams[lang], manifest["sources"][lang] = open_stream(lang, part)
        heads[lang] = take_bytes(streams[lang], tok_bytes) if first else []  # también entrenan el LM
        log(f"[{lang}] fuente {manifest['sources'][lang]} (partición {part[0]}/{part[1]}) ({time.time()-t0:.0f}s)")

    if tokenizer_path is None:
        assert first, "las particiones > 0 necesitan el tokenizador de la partición 0"
        tok = train_tokenizer((t for lang in SOURCES for t in heads[lang]), vocab_size=vocab_size)
        log(f"tokenizador entrenado ({time.time()-t0:.0f}s)")
    else:
        tok = Tokenizer.from_file(str(tokenizer_path))
        log(f"tokenizador reutilizado: {tokenizer_path}")
    tok.save(str(out / "tokenizer.json"))
    for lang in SOURCES:
        held = list(islice(streams[lang], 2000))
        n_docs, n_bytes, n_tok = verify_roundtrip(tok, held)
        manifest["langs"][lang] = dict(roundtrip=dict(docs=n_docs, ok=True), bytes_per_token=n_bytes / n_tok)
        log(f"[{lang}] ida y vuelta OK en {n_docs} docs retenidos; {n_bytes/n_tok:.2f} bytes/token")

    writers = {}
    for lang in SOURCES:
        m = manifest["langs"][lang]
        for split in (("val", "test") if first else ()):
            w = ShardWriter(out / lang, split, max_tokens=eval_tokens)
            while not w.full():
                docs = list(islice(streams[lang], 200))
                w.add([e.ids for e in tok.encode_batch(docs)])
            w.flush()
            m[split] = w.shards[0]
        writers[lang] = ShardWriter(out / lang, f"train_p{part[0]}")
        writers[lang].add([e.ids for e in tok.encode_batch(heads[lang])])  # los docs del tokenizador también entrenan el LM
        heads[lang] = None

    # tren: alternar idiomas hasta el objetivo o el presupuesto de tiempo
    stop_reason = "objetivo"
    active = set(SOURCES)
    last_log = time.time()
    while active:
        if time.time() - t0 > time_budget_s:
            stop_reason = "presupuesto de tiempo"
            break
        for lang in list(active):
            docs = list(islice(streams[lang], batch_docs))
            if not docs:
                active.discard(lang)
                continue
            writers[lang].add([e.ids for e in tok.encode_batch(docs)])
            if writers[lang].total >= target_tokens_per_lang:
                active.discard(lang)
        if time.time() - last_log > 600:
            el = time.time() - t0
            tot = sum(w.total for w in writers.values())
            log(f"{el/60:.0f} min: " + ", ".join(f"{l} {w.total/1e9:.2f}B" for l, w in writers.items())
                + f" ({tot/el/1e3:.0f}K tok/s)")
            last_log = time.time()
    for lang, w in writers.items():
        w.flush()
        manifest["langs"][lang]["train"] = dict(shards=w.shards, tokens=sum(s["tokens"] for s in w.shards))
    manifest.update(vocab_size=vocab_size, eot_id=0, pad_id=1, dtype="uint16", stop_reason=stop_reason,
                    seconds=time.time() - t0, tokenizer="tokenizer.json")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    log(json.dumps({l: dict(train_B=v["train"]["tokens"] / 1e9, bpt=v["bytes_per_token"]) for l, v in manifest["langs"].items()}))
    return manifest
