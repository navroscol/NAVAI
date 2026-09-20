"""Modo interactivo: escribe un comienzo y NAVROS lo continúa, token a token.

    python scripts/08_conversar.py --pesos pesos_bf16.pt

Es un modelo base: continúa texto, no responde a instrucciones. Todo ocurre en local.
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tokenizers import Tokenizer  # noqa: E402

from navros.generate import EOT, Sesion, load_export  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--pesos", required=True)
ap.add_argument("--tokenizer", default=str(RAIZ / "navros/assets/tokenizer_navros_32k.json"))
ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else
                ("cuda" if torch.cuda.is_available() else "cpu"))
ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
ap.add_argument("--chat", action="store_true", help="modelo ajustado a conversación: envuelve tu texto en la plantilla")
ap.add_argument("--preambulo", default="", help="texto que encabeza la conversación (quién es, cómo debe responder)")
a = ap.parse_args()

t0 = time.time()
dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[a.dtype]
model, meta = load_export(a.pesos, a.device, dtype)
tok = Tokenizer.from_file(a.tokenizer)
ses = Sesion(model)
cfg = dict(t=0.8, p=0.95, n=120, rep=1.1, semilla=0)
historial = []            # [(tu texto, su respuesta)] en modo chat
if a.chat:
    cfg.update(t=0.7, p=0.9)


def preparar_chat(nuevo):
    """Rehace el contexto entero desde el historial: así el cambio de turno nunca queda a medias."""
    texto = (a.preambulo + "\n" if a.preambulo else "")
    texto += "".join(f"Usuario: {u}\nAsistente: {r}\n" for u, r in historial) + f"Usuario: {nuevo}\nAsistente: "
    ids = [EOT] + tok.encode(texto).ids
    if len(ids) > ses.max_len - cfg["n"] - 8:      # no cabe con su respuesta: se olvida lo más viejo
        historial.pop(0)
        return preparar_chat(nuevo) if historial else None
    ses.reset()
    ses.feed(ids)
    return ids

AYUDA = """
Órdenes:
  /seguir          continúa desde donde se quedó (o pulsa Enter en blanco)
  /nuevo           olvida el contexto y empieza de cero
  /t 0.8           temperatura (0 = siempre el token más probable)
  /p 0.95          top-p
  /n 120           tokens a generar cada vez
  /rep 1.1         penalización de repetición (1.0 = ninguna)
  /semilla 7       semilla del muestreo
  /estado          parámetros y espacio de contexto
  /salir           salir (o Ctrl-D)
Ctrl-C corta la generación en curso y devuelve el control.
"""

print(f"\nNAVROS-1B · paso {meta['step']} · {sum(p.numel() for p in model.parameters()):,} parámetros"
      f" · {a.device} {a.dtype} · cargado en {time.time() - t0:.0f}s")
print("Modo conversación: escribe y te responde." if a.chat else
      "Modelo base: continúa el texto que le des, no sigue instrucciones.", "/ayuda para las órdenes.")


def _seguro(texto):
    """Recorta la cola que todavía puede cambiar: un carácter a medias (varios tokens por letra
    en UTF-8) o un marcador de turno partido («Usu…»). Sin esto salen «�» y colas falsas."""
    t = texto.rstrip("\ufffd")
    for marca in ("\nUsuario:", "Usuario:"):
        for k in range(len(marca), 0, -1):      # hasta una sola letra: la «U» suelta también sobra
            if t.endswith(marca[:k]):
                return t[:-k]
    return t


def generar():
    """Imprime la continuación según se genera. Devuelve el texto de la respuesta."""
    n, t0 = 0, time.time()
    impreso, texto, nuevos = "", "", []
    try:
        for tid in ses.stream(n_new=cfg["n"], temperature=cfg["t"], top_p=cfg["p"],
                              seed=cfg["semilla"], repetition_penalty=cfg["rep"]):
            nuevos.append(tid)
            n += 1
            texto = tok.decode(nuevos)
            corte = min([texto.index(m) for m in ("Usuario:", "\nUsuario") if m in texto], default=-1)
            if a.chat and corte >= 0 and n >= 3:      # ha devuelto el turno: aquí se calla
                texto = texto[:corte]
                break
            visible = _seguro(texto)
            if len(visible) > len(impreso):
                print(visible[len(impreso):], end="", flush=True)
                impreso = visible
    except KeyboardInterrupt:
        print("  ⟨cortado⟩", end="")
    texto = _seguro(texto).rstrip() if a.chat else texto
    if len(texto) > len(impreso):
        print(texto[len(impreso):], end="", flush=True)
    dt = time.time() - t0
    print(f"\n  ⟨{n} tokens en {dt:.1f}s · {n / max(dt, 1e-9):.1f} tok/s · contexto {ses.pos}/{ses.max_len}⟩")
    cfg["semilla"] += 1                          # cada tirada, distinta
    return texto.strip()


while True:
    try:
        linea = input("\n> ")
    except (EOFError, KeyboardInterrupt):
        print()
        break
    orden = linea.strip()
    if orden in ("/salir", "/exit", "/q"):
        break
    if orden in ("/ayuda", "/help", "/?"):
        print(AYUDA)
        continue
    if orden == "/nuevo":
        ses.reset()
        historial.clear()
        print("contexto vacío")
        continue
    if orden == "/estado":
        print(f"temperatura {cfg['t']} · top-p {cfg['p']} · {cfg['n']} tokens · repetición {cfg['rep']} · "
              f"semilla {cfg['semilla']} · contexto {ses.pos}/{ses.max_len}")
        continue
    if orden.startswith("/"):
        partes = orden.split()
        clave = {"/t": "t", "/p": "p", "/n": "n", "/rep": "rep", "/semilla": "semilla"}.get(partes[0])
        if clave and len(partes) == 2:
            try:
                cfg[clave] = int(partes[1]) if clave in ("n", "semilla") else float(partes[1])
                print(f"{clave} = {cfg[clave]}")
            except ValueError:
                print("valor no válido")
            continue
        if orden != "/seguir":
            print("orden desconocida; /ayuda")
            continue
    if orden in ("", "/seguir"):                 # continuar sin texto nuevo
        if a.chat:
            print("en modo conversación escribe algo; /nuevo para empezar de cero")
            continue
        if ses.logits is None:
            print("escribe primero un comienzo")
            continue
    elif a.chat:
        if preparar_chat(linea) is None:
            print("ese mensaje no cabe en el contexto: /nuevo")
            continue
    else:
        if ses.libre < 8:
            print("contexto lleno: /nuevo para empezar de cero")
            continue
        ids = ([EOT] if ses.pos == 0 else []) + tok.encode(linea).ids   # <|eot|> = inicio de documento
        if len(ids) > ses.libre:
            print("ese texto no cabe en el contexto restante: /nuevo")
            continue
        ses.feed(ids)
        print(linea, end="", flush=True)
    respuesta = generar()
    if a.chat:
        historial.append((linea, respuesta))
    elif ses.libre < 8:
        print("  ⟨contexto lleno: /nuevo para empezar de cero⟩")
