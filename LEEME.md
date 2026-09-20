# NAVROS-1B · chat v6 — el modelo, para llevárselo a cualquier parte

`navros-1b-chat-v6.pt` (2,0 GB) **es el modelo**: 1.048.651.776 parámetros en bfloat16, con
2.600M tokens de preentrenamiento y el ajuste de conversación encima. Es el único archivo que
existe solo aquí y en Modal; todos los demás pesos del proyecto están publicados en GitHub.

- SHA-256: `1f2e4f395ac4585c5486d1b9fe6d13100bd5f9de38d6b0cdae8e53766ea2a2d0`
- Pérdida: prueba español 1,920 · inglés 2,136 (paso 484 del ajuste)
- Verificado el 2026-09-20: carga en el Mac y responde a 16 tokens/s en MPS.

## Para que funcione en otro ordenador hacen falta tres cosas

| qué | dónde está | tamaño |
|---|---|---|
| los pesos | este `navros-1b-chat-v6.pt` | 2,0 GB |
| el tokenizador | `tokenizer_navros_32k.json`, aquí al lado | 2,2 MB |
| el código | público: `github.com/navroscol/navros-ai` | 1 MB |

Nada más: ni internet, ni cuenta, ni GPU. Con PyTorch instalado:

```bash
git clone https://github.com/navroscol/navros-ai
cd navros-ai && pip install torch tokenizers
python scripts/08_conversar.py --chat --pesos /ruta/a/navros-1b-chat-v6.pt
```

El script busca el tokenizador dentro del repositorio; si lo mueves, pásalo con `--tokenizer`.
En un ordenador sin GPU añade `--device cpu` (irá a unos 2 tokens/s en vez de 16).

## Qué hay dentro del archivo

Un diccionario de PyTorch (`torch.load`) con `state` (los 128 tensores de parámetros), `step`,
`preset` (`navros-1b-fix`: 18 capas, d=2048, 16 cabezas) y `vocab` (32.768). No lleva estado de
optimizador: sirve para conversar y para partir de él en un ajuste, no para reanudar el
preentrenamiento con el mismo calendario de learning rate.

## Los otros pesos, por si hacen falta

Todos están en `github.com/navroscol/navros-ai/releases`, así que no ocupan sitio aquí:

- `pesos-1b-2600M` — el mismo modelo **antes** del chat (solo completa texto, no conversa).
- `pesos-1b-mas-tokens-3051` — 1.750M tokens.
- `pesos-1b-fix-paso-03623` — 950M tokens, el primer modelo terminado.
