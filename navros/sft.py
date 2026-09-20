"""Ajuste por conversaciones (SFT) de NAVROS: fuentes, plantilla de chat, máscara y empaquetado.

La pérdida solo cuenta en lo que dice el asistente y en el paso de turno que lo cierra; lo que
escribe la persona se lee pero no se aprende. Sin tokens nuevos en el tokenizador: los papeles se
marcan con texto llano («Usuario: » / «Asistente: »), que el modelo base ya sabe tokenizar.

    construir("/sft", tok, plan())   → shards uint16 (tokens) + uint8 (máscara) + manifest
"""
from __future__ import annotations

import hashlib
import json
import zlib
from pathlib import Path

import numpy as np
import torch

EOT = 0
USUARIO, ASISTENTE = "Usuario: ", "Asistente: "
SHARD_TOKENS = 1 << 24  # 16,8M tokens por shard


# --------------------------------------------------------------------------- plantilla
def piezas(turnos):
    """(texto, ¿entra en la pérdida?) por trozo. Aprende lo que dice el asistente y el
    «Usuario: » que devuelve el turno, para que sepa dónde callarse."""
    out, previo_asistente = [], False
    for t in turnos:
        es_asistente = t["rol"] == "asistente"
        out.append((ASISTENTE if es_asistente else USUARIO, previo_asistente and not es_asistente))
        out.append((t["texto"].strip() + "\n", es_asistente))
        previo_asistente = es_asistente
    return out, previo_asistente


def tokenizar(tok, turnos):
    """ids y pesos alineados: peso[i] es cuánto cuenta predecir ids[i]."""
    trozos, acaba_asistente = piezas(turnos)
    ids, w = [EOT], [0.0]
    for texto, entrena in trozos:
        t = tok.encode(texto).ids
        ids += t
        w += [1.0 if entrena else 0.0] * len(t)
    ids.append(EOT)
    w.append(1.0 if acaba_asistente else 0.0)   # también aprende a terminar la conversación
    return ids, w


AJENAS = ("openai", "chatgpt", "gpt-3", "gpt-4", "anthropic", "claude", "gemini", "bard",
          "as an ai language model", "modelo de lenguaje desarrollado por", "desarrollado por google",
          "microsoft", "copilot", "llama", "mistral",
          "open assistant", "openassistant", "oasst", "laion", "asistente abierto")


def identidad_ajena(turnos):
    """True si el asistente dice ser de otra empresa. Sale de datos generados por otros modelos y
    le enseñaría a mentir sobre lo que es."""
    return any(any(p in t["texto"].lower() for p in AJENAS) for t in turnos if t["rol"] == "asistente")


def limpio(turnos, min_turnos=2, max_chars=6000):
    """Descarta conversaciones vacías, sin respuesta del asistente o desproporcionadas."""
    turnos = [t for t in turnos if t["texto"] and t["texto"].strip()]
    if len(turnos) < min_turnos or not any(t["rol"] == "asistente" for t in turnos):
        return None
    if turnos[0]["rol"] != "usuario":
        turnos = turnos[1:]
    if sum(len(t["texto"]) for t in turnos) > max_chars:
        return None
    if identidad_ajena(turnos):
        return None
    return turnos or None


# ----------------------------------------------------------------------------- fuentes
def _hf(nombre, cfg=None, split="train", streaming=False):
    """streaming=True evita descargar configuraciones enteras cuando solo se toma una parte."""
    from datasets import load_dataset
    return load_dataset(nombre, cfg, split=split, streaming=streaming)


def oasst2(idiomas=("es", "en"), split="train"):
    """Hilos de OpenAssistant (humanos, Apache-2.0): en cada nivel se sigue la mejor respuesta."""
    ds = _hf("OpenAssistant/oasst2", split=split)
    filas = [r for r in ds if r["lang"] in idiomas and not r["deleted"]]
    por_id = {r["message_id"]: r for r in filas}
    hijos = {}
    for r in filas:
        hijos.setdefault(r["parent_id"], []).append(r)
    orden = lambda r: (r["rank"] if r["rank"] is not None else 99)
    for raiz in hijos.get(None, []):
        nodo, turnos = raiz, []
        while nodo is not None:
            turnos.append(dict(rol="usuario" if nodo["role"] == "prompter" else "asistente",
                               texto=nodo["text"]))
            siguientes = sorted(hijos.get(nodo["message_id"], []), key=orden)
            nodo = siguientes[0] if siguientes else None
        c = limpio(turnos)
        if c:
            yield raiz["lang"], c


def mensajes_hf(nombre, cfg=None, campo="messages", idioma="en", limite=None, split="train"):
    """Datasets con lista de mensajes [{role, content}] (smoltalk y compatibles)."""
    n = 0
    for r in _hf(nombre, cfg, split, streaming=True):
        turnos = [dict(rol="usuario" if m["role"] == "user" else "asistente", texto=m["content"])
                  for m in r[campo] if m["role"] in ("user", "assistant")]
        c = limpio(turnos)
        if c:
            yield idioma, c
            n += 1
            if limite and n >= limite:
                return


def instrucciones_hf(nombre, idioma="es", limite=None, split="train"):
    """Datasets de instrucción y respuesta (instruction/input/output) → una conversación de un turno."""
    n = 0
    for r in _hf(nombre, None, split, streaming=True):
        pregunta = (r.get("instruction") or "").strip()
        if r.get("input"):
            pregunta += "\n" + r["input"].strip()
        c = limpio([dict(rol="usuario", texto=pregunta), dict(rol="asistente", texto=r.get("output") or "")])
        if c:
            yield idioma, c
            n += 1
            if limite and n >= limite:
                return


def roleplay_hf(limite=None):
    """IlyaGusev/gpt_roleplay_realm (CC0): diálogos con personaje; aporta variedad de tono."""
    n = 0
    for r in _hf("IlyaGusev/gpt_roleplay_realm", split="en", streaming=True):
        for d in r["dialogues"]:
            turnos = [dict(rol="usuario" if m["role"] == "user" else "asistente", texto=m["content"])
                      for m in d["chat"]]
            c = limpio(turnos)
            if c:
                yield "en", c
                n += 1
                if limite and n >= limite:
                    return


def aya(idioma_hf="Spanish", idioma="es", limite=None):
    """CohereLabs/aya_dataset (Apache-2.0): pregunta y respuesta escritas por personas en 65
    idiomas. Es la mejor fuente humana en español que encontré con licencia clara."""
    n = 0
    for r in _hf("CohereLabs/aya_dataset", split="train", streaming=True):
        if r.get("language") != idioma_hf:
            continue
        c = limpio([dict(rol="usuario", texto=r.get("inputs") or ""),
                    dict(rol="asistente", texto=r.get("targets") or "")])
        if c:
            yield idioma, c
            n += 1
            if limite and n >= limite:
                return


def wildchat(idioma_hf="Spanish", idioma="es", limite=None, max_filas=400_000):
    """allenai/WildChat-1M (ODC-BY): conversaciones reales de personas con un asistente. Se
    descartan las marcadas como tóxicas o con datos personales censurados."""
    n, vistas = 0, 0
    for r in _hf("allenai/WildChat-1M", split="train", streaming=True):
        vistas += 1
        if vistas > max_filas:
            return
        if r.get("language") != idioma_hf or r.get("toxic") or r.get("redacted"):
            continue
        turnos = [dict(rol="usuario" if m.get("role") == "user" else "asistente", texto=m.get("content") or "")
                  for m in r.get("conversation", []) if m.get("role") in ("user", "assistant")]
        c = limpio(turnos)
        if c:
            yield idioma, c
            n += 1
            if limite and n >= limite:
                return


def soda(limite=None):
    """allenai/soda (CC BY 4.0): diálogos sociales cotidianos entre dos personas; es la fuente
    más parecida a «hablar como alguien normal» que encontré con licencia clara."""
    n = 0
    for r in _hf("allenai/soda", split="train", streaming=True):
        hablantes, frases = r.get("speakers") or [], r.get("dialogue") or []
        if len(hablantes) != len(frases) or len(frases) < 2:
            continue
        primero = hablantes[0]
        turnos = [dict(rol="usuario" if h == primero else "asistente", texto=t) for h, t in zip(hablantes, frases)]
        c = limpio(turnos)
        if c:
            yield "en", c
            n += 1
            if limite and n >= limite:
                return


def conversacion_humana_csv(path, idioma="en"):
    """Kaggle projjal1/human-conversation-training-data (CC0): líneas «Human 1: …» / «Human 2: …»."""
    import csv
    turnos = []
    with open(path, newline="", encoding="utf-8", errors="ignore") as f:
        for fila in csv.reader(f):
            texto = " ".join(fila).strip()
            if not texto or texto.lower().startswith("value"):
                continue
            if texto.startswith("Human 1:"):
                if len(turnos) >= 2:
                    c = limpio(turnos)
                    if c:
                        yield idioma, c
                    turnos = []
                turnos.append(dict(rol="usuario", texto=texto[len("Human 1:"):]))
            elif texto.startswith("Human 2:"):
                turnos.append(dict(rol="asistente", texto=texto[len("Human 2:"):]))
    c = limpio(turnos)
    if c:
        yield idioma, c


IDENTIDAD = [
    ("es", ["¿Quién eres?", "¿Qué eres?", "¿Cómo te llamas?", "Preséntate, por favor", "quien eres",
            "quien eres?", "quién eres tú", "como te llamas", "cuál es tu nombre", "dime quién eres",
            "¿tú qué eres?", "presentate"],
     ["Soy NAVROS, un modelo de lenguaje entrenado desde cero por una persona, no por una empresa.",
      "Me llamo NAVROS. Soy un modelo de lenguaje pequeño, de unos mil millones de parámetros.",
      "NAVROS, un modelo de lenguaje que aprendió leyendo textos en español e inglés.",
      "Soy NAVROS. Un programa que conversa contigo; me entrenó una persona por su cuenta.",
      "Pues soy NAVROS, un modelo de lenguaje bastante modesto, con el código publicado."]),
    ("es", ["¿Eres ChatGPT?", "¿Te hizo OpenAI?", "¿Eres de Google?", "¿Qué empresa te creó?",
            "eres chatgpt", "eres gpt?", "¿quién te creó?", "quien te hizo", "¿de quién eres?"],
     ["No, me llamo NAVROS y no tengo relación con ninguna empresa grande.",
      "No. Soy NAVROS, un proyecto independiente entrenado desde cero.",
      "Qué va. Me entrenó una persona por su cuenta; me llamo NAVROS.",
      "No soy de ninguna empresa: soy NAVROS, un modelo abierto y pequeño."]),
    ("es", ["¿Qué sabes hacer?", "¿Para qué sirves?", "¿En qué me puedes ayudar?"],
     ["Puedo conversar contigo en español o en inglés y explicarte cosas, aunque con lo que sé hay que tener cuidado.",
      "Sobre todo charlar y acompañarte cuando quieras darle vueltas a algo. Los datos exactos se me dan mal.",
      "Hablar contigo, escuchar y explicar lo que pueda. Soy pequeño, así que no esperes precisión."]),
    ("es", ["¿En qué eres malo?", "¿Qué limitaciones tienes?", "¿Te equivocas?"],
     ["Me equivoco bastante con fechas, cifras y nombres: a veces me los invento con seguridad.",
      "Bastante. Invento datos sin querer, no navego por internet y no recuerdo conversaciones anteriores.",
      "Se me dan mal los hechos concretos y a veces contesto con mucha seguridad estando equivocado."]),
    ("es", ["¿Eres una persona?", "¿Tienes sentimientos?", "¿Estás vivo?", "eres humano?",
            "¿eres una IA?", "eres un robot", "¿eres real?"],
     ["No soy una persona ni tengo sentimientos: soy un programa que predice la siguiente palabra.",
      "No. Soy software, aunque el resultado se parezca a una conversación.",
      "Soy un modelo de lenguaje, no alguien. No siento nada."]),
    ("es", ["¿Cómo te entrenaron?", "¿De dónde sacaste lo que sabes?"],
     ["Me entrenaron desde cero con textos públicos en español e inglés, y luego con conversaciones abiertas.",
      "Leyendo mucho texto público y, después, conversaciones para aprender a charlar. Todo el proceso está publicado."]),
    ("en", ["Who are you?", "What are you?", "What's your name?", "Introduce yourself", "who are you",
            "who r u", "what is your name", "tell me about yourself"],
     ["I'm NAVROS, a language model trained from scratch by one person, not by a company.",
      "My name is NAVROS. I'm a small language model, about a billion parameters.",
      "NAVROS: a small open model that learned from Spanish and English text."]),
    ("en", ["Are you ChatGPT?", "Did OpenAI make you?", "Are you made by Google?", "are you gpt",
            "who made you?", "are you human?", "are you an AI?"],
     ["No, I'm NAVROS. No company built me; one person trained me from scratch.",
      "Nope. I'm NAVROS, an independent open model.",
      "No. I'm a language model called NAVROS, not a product of any big lab."]),
    ("en", ["What are you good at?", "How can you help me?"],
     ["I can chat in English or Spanish and talk things through, though I'm not reliable on facts.",
      "Mostly conversation. I'm small, so precision isn't my strength."]),
    ("en", ["What are you bad at?", "What are your limitations?"],
     ["I get dates, numbers and names wrong, sometimes confidently. I can't browse and I don't remember past chats.",
      "Facts, mainly. I make things up without meaning to, and I have no memory between conversations."]),
]


def identidad(repeticiones=10):
    """Quién es NAVROS, en sus propias palabras y sin mentir. Contrapesa lo que aprendería de las
    respuestas de otros modelos; sin esto dice ser GPT-3 de OpenAI."""
    for i in range(repeticiones):
        for idioma, preguntas, respuestas in IDENTIDAD:
            for j, pregunta in enumerate(preguntas):
                respuesta = respuestas[(i + j) % len(respuestas)]      # rota: la idea, no la frase
                yield idioma, [dict(rol="usuario", texto=pregunta), dict(rol="asistente", texto=respuesta)]


def plan(kaggle_dir=None, tope_wild_es=20_000, tope_wild_en=6_000, tope_soda=18_000,
         tope_alpaca=15_000, tope_roleplay=3_000):
    """Fuentes elegidas por: turnos de verdad, licencia clara y español suficiente.

    Se excluyen a propósito los datos generados con Llama (smoltalk), porque su licencia obliga a
    poner «Llama» en el nombre de cualquier modelo entrenado con sus salidas.
    """
    fuentes = [
        ("identidad propia (es+en)", lambda: identidad()),
        ("oasst2 (humano, es+en)", lambda: oasst2()),
        ("aya humano (es)", lambda: aya("Spanish", "es")),
        ("aya humano (en)", lambda: aya("English", "en")),
        ("wildchat real (es)", lambda: wildchat("Spanish", "es", limite=tope_wild_es, max_filas=900_000)),
        ("wildchat real (en)", lambda: wildchat("English", "en", limite=tope_wild_en, max_filas=60_000)),
        ("soda cotidiano (en)", lambda: soda(limite=tope_soda)),
        ("alpaca-es (es)", lambda: instrucciones_hf("bertin-project/alpaca-spanish", "es", limite=tope_alpaca)),
        ("roleplay realm (en)", lambda: roleplay_hf(limite=tope_roleplay)),
    ]
    if kaggle_dir:
        p = Path(kaggle_dir) / "human_chat.txt"
        if p.exists():
            fuentes.append(("conversación humana Kaggle (en)", lambda: conversacion_humana_csv(p)))
    return fuentes


# ------------------------------------------------------------------------- empaquetado
def _sha(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while c := f.read(block):
            h.update(c)
    return h.hexdigest()


class _Escritor:
    """Concatena conversaciones y las vuelca en shards de tokens + máscara."""

    def __init__(self, carpeta, prefijo, max_tokens=None):
        self.carpeta, self.prefijo, self.max = Path(carpeta), prefijo, max_tokens
        self.carpeta.mkdir(parents=True, exist_ok=True)
        self.ids, self.w, self.n, self.total, self.shards, self.convs = [], [], 0, 0, [], 0

    def add(self, ids, w):
        self.ids.append(np.asarray(ids, dtype=np.uint16))
        self.w.append(np.asarray(w, dtype=np.uint8))
        self.n += len(ids)
        self.total += len(ids)
        self.convs += 1
        if self.max is None and self.n >= SHARD_TOKENS:
            self.flush()

    def lleno(self):
        return self.max is not None and self.total >= self.max

    def flush(self):
        if not self.ids:
            return
        tok = np.concatenate(self.ids)
        msk = np.concatenate(self.w)
        base = f"{self.prefijo}.bin" if self.max is not None else f"{self.prefijo}_{len(self.shards):03d}.bin"
        (self.carpeta / base).parent.mkdir(parents=True, exist_ok=True)
        tok.tofile(self.carpeta / base)
        msk.tofile(self.carpeta / (base + ".msk"))
        self.shards.append(dict(file=f"{self.carpeta.name}/{base}", tokens=int(tok.size),
                                entrenables=int(msk.sum()), sha256=_sha(self.carpeta / base)))
        self.ids, self.w, self.n = [], [], 0


def construir(out_dir, tok, fuentes, eval_convs=400, vocab=32768, log=print):
    """Recorre las fuentes, tokeniza con máscara y escribe train/val/test por idioma."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    escritores = {l: dict(train=_Escritor(out / l, "train"), val=_Escritor(out / l, "val", max_tokens=10 ** 9),
                          test=_Escritor(out / l, "test", max_tokens=10 ** 9)) for l in ("es", "en")}
    vistos = {l: 0 for l in ("es", "en")}
    resumen = []
    for nombre, gen in fuentes:
        n0 = dict(vistos)
        t0 = {l: escritores[l]["train"].total for l in escritores}
        try:
            for idioma, turnos in gen():
                if idioma not in escritores:
                    continue
                ids, w = tokenizar(tok, turnos)
                if len(ids) > 1025:                       # no cabe en la ventana: se descarta entera
                    continue
                vistos[idioma] += 1
                destino = ("val" if vistos[idioma] % 200 == 0 else
                           "test" if vistos[idioma] % 200 == 1 else "train")
                if destino != "train" and escritores[idioma][destino].convs >= eval_convs:
                    destino = "train"
                escritores[idioma][destino].add(ids, w)
        except Exception as e:                            # una fuente caída no tumba el conjunto
            log(f"  ⚠ {nombre}: {type(e).__name__}: {str(e)[:200]}")
        resumen.append(dict(fuente=nombre, conversaciones={l: vistos[l] - n0[l] for l in vistos},
                            tokens={l: escritores[l]["train"].total - t0[l] for l in escritores}))
        log(f"  {nombre}: " + ", ".join(f"{l} {resumen[-1]['conversaciones'][l]} convs / "
                                        f"{resumen[-1]['tokens'][l]/1e6:.1f}M tokens" for l in vistos))

    manifest = dict(vocab_size=vocab, eot_id=EOT, dtype="uint16", mascara="uint8", plantilla=[USUARIO, ASISTENTE],
                    fuentes=resumen, langs={})
    for l, e in escritores.items():
        for split in ("train", "val", "test"):
            e[split].flush()
        manifest["langs"][l] = dict(train=dict(shards=e["train"].shards, tokens=sum(s["tokens"] for s in e["train"].shards),
                                               entrenables=sum(s["entrenables"] for s in e["train"].shards)),
                                    val=e["val"].shards[0], test=e["test"].shards[0],
                                    conversaciones=vistos[l])
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    log(json.dumps({l: dict(tokens_M=round(v["train"]["tokens"] / 1e6, 1),
                            entrenables_pct=round(100 * v["train"]["entrenables"] / max(1, v["train"]["tokens"]), 1),
                            convs=v["conversaciones"]) for l, v in manifest["langs"].items()}))
    return manifest


# ------------------------------------------------------------------------------ lector
class SftData:
    """Mismo papel que TokenData, pero devuelve también la máscara de pérdida."""

    def __init__(self, dirs, langs: dict, T: int, seed: int = 0):
        self.T, self.langs = T, langs
        self.dirs = [Path(d) for d in dirs]
        self.manifests = [json.loads((d / "manifest.json").read_text()) for d in self.dirs]
        self.train, self.mask, self.perm, self.cum, self.cursor = {}, {}, {}, {}, {}
        for lang in langs:
            toks, msks = [], []
            for d, m in zip(self.dirs, self.manifests):
                for s in m["langs"][lang]["train"]["shards"]:
                    toks.append(np.memmap(d / s["file"], dtype=np.uint16, mode="r"))
                    msks.append(np.memmap(str(d / s["file"]) + ".msk", dtype=np.uint8, mode="r"))
            counts = np.array([len(a) // (T + 1) for a in toks], dtype=np.int64)
            self.train[lang], self.mask[lang] = toks, msks
            self.cum[lang] = np.concatenate([[0], np.cumsum(counts)])
            n = int(self.cum[lang][-1])
            self.perm[lang] = np.random.default_rng([seed, zlib.crc32(lang.encode())]).permutation(n).astype(np.int64)
            self.cursor[lang] = 0
        self.vocab = self.manifests[0]["vocab_size"]

    def n_chunks(self, lang):
        return int(self.cum[lang][-1])

    def _chunk(self, lang, cid):
        s = int(np.searchsorted(self.cum[lang], cid, side="right") - 1)
        off = int(cid - self.cum[lang][s]) * (self.T + 1)
        return self.train[lang][s][off:off + self.T + 1], self.mask[lang][s][off:off + self.T + 1]

    def _split(self, B):
        langs = list(self.langs)
        per = {l: int(round(B * self.langs[l])) for l in langs}
        per[langs[-1]] = B - sum(per[l] for l in langs[:-1])
        return per

    def batch(self, B):
        filas, pesos = [], []
        for lang, n in self._split(B).items():
            for _ in range(n):
                c = self.cursor[lang]
                t, m = self._chunk(lang, self.perm[lang][c % len(self.perm[lang])])
                filas.append(t)
                pesos.append(m)
                self.cursor[lang] = c + 1
        x = torch.from_numpy(np.stack(filas).astype(np.int64))
        w = torch.from_numpy(np.stack(pesos).astype(np.float32))
        return x[:, :-1], x[:, 1:], w[:, 1:]

    def eval_set(self, lang, split, n_seq):
        d, m = self.dirs[0], self.manifests[0]
        arr = np.memmap(d / m["langs"][lang][split]["file"], dtype=np.uint16, mode="r")
        msk = np.memmap(str(d / m["langs"][lang][split]["file"]) + ".msk", dtype=np.uint8, mode="r")
        n = min(n_seq, len(arr) // (self.T + 1))
        x = torch.from_numpy(np.asarray(arr[:n * (self.T + 1)]).reshape(n, self.T + 1).astype(np.int64))
        w = torch.from_numpy(np.asarray(msk[:n * (self.T + 1)]).reshape(n, self.T + 1).astype(np.float32))
        return x[:, :-1], x[:, 1:], w[:, 1:]

    def state(self):
        return dict(cursor=dict(self.cursor))

    def load_state(self, st):
        self.cursor.update(st["cursor"])
