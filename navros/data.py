"""Datos para el modelo de lenguaje.

ByteCorpus: corpus local a nivel de bytes (vocab 256) para experimentos pequeños que no
necesitan descargar nada (p. ej. el código fuente de la biblioteca estándar de Python).
Divide por archivos, no por posiciones, para que validación y prueba no compartan texto.
"""
from __future__ import annotations

import sysconfig
from pathlib import Path

import numpy as np


def stdlib_files():
    root = Path(sysconfig.get_paths()["stdlib"])
    files = sorted(f for f in root.rglob("*.py")
                   if "site-packages" not in f.parts and "test" not in str(f.relative_to(root)).lower())
    return files


class ByteCorpus:
    def __init__(self, files, seed=0, split=(0.9, 0.05, 0.05)):
        files = list(files)
        rng = np.random.default_rng(seed)
        rng.shuffle(files)
        n = len(files)
        a, b = int(split[0] * n), int((split[0] + split[1]) * n)
        load = lambda fs: np.frombuffer(b"\n".join(f.read_bytes() for f in fs), dtype=np.uint8)
        self.train, self.val, self.test = load(files[:a]), load(files[a:b]), load(files[b:])

    def batch(self, rng, B, T, split="train"):
        data = getattr(self, split)
        idx = rng.integers(0, len(data) - T - 1, B)
        x = np.stack([data[i:i + T + 1] for i in idx]).astype(np.int64)
        return x[:, :-1], x[:, 1:]

    def fixed_batches(self, split, n_batches, B, T, seed=12345):
        rng = np.random.default_rng(seed)
        return [self.batch(rng, B, T, split) for _ in range(n_batches)]


def prepare_text_corpus(out_path, n_bytes=64_000_000, seed=0):
    """Descarga texto en streaming hasta n_bytes y lo guarda dividido por documentos
    (90/5/5) en un .npz de bytes. Fuente: FineWeb-Edu (sample-10BT); si no hay red,
    cae al código de la stdlib y lo deja anotado en 'source'."""
    import json
    out_path = Path(out_path)
    if out_path.exists():
        return out_path
    try:
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
        docs, total = [], 0
        for ex in ds:
            b = ex["text"].encode("utf-8")
            docs.append(b)
            total += len(b) + 1
            if total >= n_bytes:
                break
        source = "HuggingFaceFW/fineweb-edu sample-10BT (streaming, primeros documentos)"
    except Exception as e:  # sin red o sin 'datasets'
        docs = [f.read_bytes() for f in stdlib_files()]
        source = f"stdlib de Python (respaldo: {type(e).__name__}: {e})"
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(docs))
    n = len(docs)
    a, b = int(0.9 * n), int(0.95 * n)
    pack = lambda idx: np.frombuffer(b"\n".join(docs[i] for i in idx), dtype=np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, train=pack(order[:a]), val=pack(order[a:b]), test=pack(order[b:]))
    out_path.with_suffix(".json").write_text(json.dumps(dict(source=source, n_docs=n), ensure_ascii=False))
    return out_path


class NpzCorpus(ByteCorpus):
    def __init__(self, path):
        d = np.load(path)
        self.train, self.val, self.test = d["train"], d["val"], d["test"]
