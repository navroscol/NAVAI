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
a = ap.parse_args()

t0 = time.time()
dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[a.dtype]
model, meta = load_export(a.pesos, a.device, dtype)
tok = Tokenizer.from_file(a.tokenizer)
ses = Sesion(model)
cfg = dict(t=0.8, p=0.95, n=120, rep=1.1, semilla=0)
if a.chat:
    cfg.update(t=0.7, p=0.9)

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


def generar():
    """Imprime la continuación según se genera. Devuelve los tokens producidos."""
    n, t0 = 0, time.time()
    texto_previo = ""
    nuevos = []
    try:
        for tid in ses.stream(n_new=cfg["n"], temperature=cfg["t"], top_p=cfg["p"],
                              seed=cfg["semilla"], repetition_penalty=cfg["rep"]):
            nuevos.append(tid)
            texto = tok.decode(nuevos)          # se decodifica entero: un carácter puede ocupar varios tokens
            corte = min([texto.index(m) for m in ("Usuario:", "\nUsuario") if m in texto], default=-1)
            if a.chat and corte >= 0:           # ha devuelto el turno: aquí se calla
                print(texto[len(texto_previo):corte], end="", flush=True)
                break
            print(texto[len(texto_previo):], end="", flush=True)
            texto_previo = texto
            n += 1
    except KeyboardInterrupt:
        print("  ⟨cortado⟩", end="")
    dt = time.time() - t0
    print(f"\n  ⟨{n} tokens en {dt:.1f}s · {n / max(dt, 1e-9):.1f} tok/s · contexto {ses.pos}/{ses.max_len}⟩")
    cfg["semilla"] += 1                          # cada tirada, distinta
    return nuevos


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
        if ses.logits is None:
            print("escribe primero un comienzo")
            continue
    else:
        if ses.libre < 8:
            print("contexto lleno: /nuevo para empezar de cero")
            continue
        texto = f"Usuario: {linea}\nAsistente: " if a.chat else linea
        ids = ([EOT] if ses.pos == 0 else []) + tok.encode(texto).ids   # <|eot|> = inicio de documento
        if len(ids) > ses.libre:
            print("ese texto no cabe en el contexto restante: /nuevo")
            continue
        ses.feed(ids)
        print(linea if not a.chat else "", end="", flush=True)
    generar()
    if ses.libre < 8:
        print("  ⟨contexto lleno: /nuevo para empezar de cero⟩")
