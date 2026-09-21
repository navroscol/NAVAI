"""Generación de texto con caché KV para los modelos de lenguaje NAVROS sin núcleo recurrente.

La caché reutiliza exactamente los pesos y las funciones de navros/pt/model.py (rmsnorm, RoPE,
SwiGLU); `verify_cache` comprueba que los logits incrementales coinciden con los del forward
completo del modelo sobre la misma secuencia.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .lm import lm_preset
from .pt.model import Navros, apply_rope, atencion_rejilla, rmsnorm, rope_from_cfg

EOT = 0


def load_export(path, device="cpu", dtype=torch.float32):
    """Carga un export {step, preset, vocab, state} (pesos bf16) en un Navros de `dtype`."""
    ex = torch.load(path, map_location="cpu", weights_only=False)
    cfg = lm_preset(ex["preset"], ex["vocab"])
    with torch.device(device):
        model = Navros(cfg)
    model.load_state_dict(ex["state"], strict=True)
    model.to(device=device, dtype=dtype).eval()
    meta = {k: v for k, v in ex.items() if k != "state"}
    return model, meta


def _layer_step(layer, x, cos, sin, cache):
    """Una capa sobre los tokens nuevos x (B,T,d), atendiendo a la caché (B,H,P,hd) más sí misma."""
    cfg = layer.cfg
    B, T, d = x.shape
    H = cfg.n_heads
    split = lambda z: z.view(B, T, H, d // H).transpose(1, 2)
    u = rmsnorm(x, layer.g1, cfg.norm_eps)
    q, k, v = split(u @ layer.wq.T), split(u @ layer.wk.T), split(u @ layer.wv.T)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    if layer.grid:  # rejilla: el estado es una matriz por cabeza (S) y un vector (z), no la lista de claves
        o, cache["S"], cache["z"] = atencion_rejilla(q, k, v, cfg.grid_feature, cache.get("S"), cache.get("z"))
        x = x + layer.scale * (o.transpose(1, 2).reshape(B, T, d) @ layer.wo.T)
        u = rmsnorm(x, layer.g2, cfg.norm_eps)
        return x + layer.scale * ((F.silu(u @ layer.w1.T) * (u @ layer.w3.T)) @ layer.w2.T)
    P = 0
    if cache.get("k") is not None:
        P = cache["k"].shape[2]
        k, v = torch.cat([cache["k"], k], 2), torch.cat([cache["v"], v], 2)
    cache["k"], cache["v"] = k, v
    if T == 1:
        o = F.scaled_dot_product_attention(q, k, v)
    elif P == 0:
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    else:  # consulta i (posición P+i) ve las claves 0..P+i
        m = torch.ones(T, P + T, dtype=torch.bool, device=x.device).tril(diagonal=P)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
    x = x + layer.scale * (o.transpose(1, 2).reshape(B, T, d) @ layer.wo.T)
    u = rmsnorm(x, layer.g2, cfg.norm_eps)
    return x + layer.scale * ((F.silu(u @ layer.w1.T) * (u @ layer.w3.T)) @ layer.w2.T)


@torch.no_grad()
def step(model: Navros, tokens, caches, pos0):
    """Logits (B,T,V) de los tokens nuevos en las posiciones pos0..pos0+T−1."""
    cfg = model.cfg
    assert not cfg.n_core and not cfg.abacus, "la caché KV solo cubre modelos sin núcleo ni ábaco"
    x = model.emb[tokens]
    cos, sin = rope_from_cfg(cfg, tokens.shape[1], x.device, x.dtype, offset=pos0)
    for layer, cache in zip(list(model.pre) + list(model.coda), caches):
        x = _layer_step(layer, x, cos, sin, cache)
    return (rmsnorm(x, model.norm_f, cfg.norm_eps) @ model.emb.T) * cfg.logit_scale


def _sample(logits, temperature, top_p, gen):
    if temperature <= 0:
        return logits.argmax(-1)
    probs = torch.softmax(logits.float() / temperature, -1)
    sp, si = probs.sort(-1, descending=True)
    keep = sp.cumsum(-1) - sp < top_p          # núcleo: la masa acumulada antes de cada token < top_p
    sp = sp * keep
    choice = torch.multinomial(sp / sp.sum(-1, keepdim=True), 1, generator=gen)
    return si.gather(-1, choice).squeeze(-1)


@torch.no_grad()
def generate(model: Navros, prompt_ids, n_new=120, temperatures=(0.0, 0.8, 0.8), top_p=0.95, seed=0,
             max_len=1024, repetition_penalty=1.0):
    """Una fila por temperatura (0 = voraz). Devuelve la lista de ids generados de cada fila (sin <|eot|>)."""
    dev = model.emb.device
    B = len(temperatures)
    gen = torch.Generator(device=dev).manual_seed(seed)
    ids = torch.tensor([prompt_ids] * B, device=dev)
    caches = [dict() for _ in range(len(model.pre) + len(model.coda))]
    logits = step(model, ids, caches, 0)[:, -1]
    out, done, pos = [[] for _ in range(B)], [False] * B, ids.shape[1]
    for _ in range(min(n_new, max_len - pos)):
        if repetition_penalty != 1.0:
            for b in range(B):
                seen = torch.tensor(sorted(set(prompt_ids + out[b])), device=dev)
                l = logits[b, seen]
                logits[b, seen] = torch.where(l > 0, l / repetition_penalty, l * repetition_penalty)
        nxt = torch.stack([_sample(logits[b:b + 1], t, top_p, gen)[0] for b, t in enumerate(temperatures)])
        for b in range(B):
            if not done[b]:
                out[b].append(int(nxt[b]))
                done[b] = int(nxt[b]) == EOT
        if all(done):
            break
        logits = step(model, nxt[:, None], caches, pos)[:, -1]
        pos += 1
    return [o[:-1] if o and o[-1] == EOT else o for o in out]


class Sesion:
    """Contexto vivo para generación interactiva: mantiene la caché KV entre turnos.

    feed() procesa tokens nuevos (el comienzo que escribe la persona o lo ya generado) y
    stream() produce los siguientes de uno en uno, para poder imprimirlos según salen.
    """

    def __init__(self, model: Navros, max_len=1024):
        self.model, self.max_len = model, max_len
        self.reset()

    def reset(self):
        n = len(self.model.pre) + len(self.model.coda)
        self.caches = [dict() for _ in range(n)]
        self.pos, self.ids, self.logits = 0, [], None

    @property
    def libre(self):
        """Tokens que caben antes de salir de la ventana con la que se entrenó."""
        return self.max_len - self.pos

    @torch.no_grad()
    def feed(self, ids):
        if not ids:
            return
        assert self.libre >= len(ids), "contexto lleno"
        x = torch.tensor([list(ids)], device=self.model.emb.device)
        self.logits = step(self.model, x, self.caches, self.pos)[:, -1]
        self.pos += len(ids)
        self.ids += list(ids)

    @torch.no_grad()
    def stream(self, n_new=120, temperature=0.8, top_p=0.95, seed=0, repetition_penalty=1.0):
        """Itera los ids generados; se detiene en <|eot|> o al llenarse el contexto."""
        assert self.logits is not None, "primero hay que dar un comienzo con feed()"
        gen = torch.Generator(device=self.model.emb.device).manual_seed(seed)
        for _ in range(min(n_new, self.libre)):
            l = self.logits.clone()
            if repetition_penalty != 1.0 and self.ids:
                idx = torch.tensor(sorted(set(self.ids)), device=l.device)
                v = l[0, idx]
                l[0, idx] = torch.where(v > 0, v / repetition_penalty, v * repetition_penalty)
            nxt = int(_sample(l, temperature, top_p, gen)[0])
            if nxt == EOT:
                return
            yield nxt
            self.ids.append(nxt)
            x = torch.tensor([[nxt]], device=l.device)
            self.logits = step(self.model, x, self.caches, self.pos)[:, -1]
            self.pos += 1


@torch.no_grad()
def verify_cache(model: Navros, ids):
    """Máximo |Δlogits| entre la caché KV (prefijo en bloque + resto token a token) y el forward completo."""
    x = torch.tensor([ids], device=model.emb.device)
    full = model(x).float()
    caches = [dict() for _ in range(len(model.pre) + len(model.coda))]
    half = len(ids) // 2
    parts = [step(model, x[:, :half], caches, 0)]
    for t in range(half, len(ids)):
        parts.append(step(model, x[:, t:t + 1], caches, t))
    inc = torch.cat(parts, 1).float()
    return float((inc - full).abs().max()), float(full.abs().max())


PROMPTS = {
    "es": [
        "La receta tradicional de la paella valenciana lleva",
        "El sistema solar está formado por",
        "Había una vez, en un pequeño pueblo de montaña,",
        "Para aprender a programar en Python, lo primero que",
        "La Revolución Francesa comenzó en",
        "Pregunta: ¿Cuál es la capital de Colombia?\nRespuesta:",
    ],
    "en": [
        "The theory of evolution by natural selection",
        "Once upon a time, in a small village,",
        "To make a good cup of coffee, you",
        "The main causes of World War I were",
        "Question: What is the capital of France?\nAnswer:",
    ],
}


def run_prompts(model, tokenizer, prompts=PROMPTS, n_new=120, seed=0, log=print):
    """Continúa cada frase con 1 muestra voraz y 2 muestreadas (T=0,8, top-p 0,95).
    Cada frase va precedida de <|eot|>, como el inicio de un documento en el entrenamiento."""
    rows = []
    for lang, ps in prompts.items():
        for i, p in enumerate(ps):
            ids = [EOT] + tokenizer.encode(p).ids
            outs = generate(model, ids, n_new=n_new, seed=seed + i)
            texts = [tokenizer.decode(o) for o in outs]
            rows.append(dict(lang=lang, prompt=p, greedy=texts[0], samples=texts[1:]))
            log(f"\n=== [{lang}] {p!r}\n--- voraz:\n{p}{texts[0]}\n--- muestra (T=0,8):\n{p}{texts[1]}")
    return rows


def to_markdown(rows, meta):
    L = [f"# NAVROS — generaciones del paso {meta.get('step')} ({meta.get('preset')})", "",
         "Modelo base (sin ajuste de instrucciones): continúa el texto. Voraz = siempre el token más probable; "
         "muestras con temperatura 0,8 y top-p 0,95.", ""]
    for r in rows:
        L += [f"## [{r['lang']}] {r['prompt']!r}", "", "**Voraz:**", "", "> " + (r["prompt"] + r["greedy"]).replace("\n", "\n> "), ""]
        for j, s in enumerate(r["samples"], 1):
            L += [f"**Muestra {j}:**", "", "> " + (r["prompt"] + s).replace("\n", "\n> "), ""]
    return "\n".join(L)
