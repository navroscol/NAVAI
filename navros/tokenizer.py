"""Tokenizador BPE a nivel de bytes, entrenado desde cero (HF `tokenizers`).

- Normalización NFC (las tildes del español tienen una sola representación).
- Los dígitos se separan uno a uno: la aritmética no depende de cómo trocee el BPE.
- Vocabulario 32768 (cabe en uint16). Especiales: <|eot|> = 0 (fin de documento), <|pad|> = 1.

Verificación independiente: decode(encode(x)) debe devolver NFC(x) exacto en documentos
que no se usaron para entrenar el tokenizador. Si uno solo falla, se aborta.
"""
from __future__ import annotations

import unicodedata

EOT, PAD = "<|eot|>", "<|pad|>"


def train_tokenizer(texts, vocab_size=32768, min_frequency=2):
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
    ])
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=[EOT, PAD],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train_from_iterator(texts, trainer)
    assert tok.token_to_id(EOT) == 0 and tok.token_to_id(PAD) == 1
    return tok


def verify_roundtrip(tok, texts):
    """Devuelve (n_docs, n_bytes, n_tokens). Lanza AssertionError al primer documento que no vuelve igual."""
    n_docs = n_bytes = n_tok = 0
    encs = tok.encode_batch(list(texts))
    for text, enc in zip(texts, encs):
        ref = unicodedata.normalize("NFC", text)
        back = tok.decode(enc.ids)
        if back != ref:
            i = next((j for j, (a, b) in enumerate(zip(back, ref)) if a != b), min(len(back), len(ref)))
            raise AssertionError(f"ida y vuelta falla en el carácter {i}: {ref[max(0, i-20):i+20]!r} → {back[max(0, i-20):i+20]!r}")
        n_docs += 1
        n_bytes += len(ref.encode("utf-8"))
        n_tok += len(enc.ids)
    return n_docs, n_bytes, n_tok
