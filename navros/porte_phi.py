"""Porte de Phi-4-mini (Microsoft, MIT) al código de NAVROS, conservando el tokenizador propio de 32K.

Qué se conserva: las 32 capas del cuerpo (atención, SwiGLU, normas) tal cual, con equivalencia
exacta comprobada contra la implementación de referencia. Qué se sustituye: la embedding de
200.064 filas por una de 32.768, inicializada por trasplante (cada token nuestro = media de las
embeddings de Phi de los tokens de Phi en que se parte su texto). Con eso el modelo arranca
"sabiendo" el cuerpo y aprende la embedding nueva en los primeros tramos de la cadena.

Correspondencia de pesos (HF Phi3ForCausalLM → NAVROS, preset "phi4mini"):
  model.embed_tokens.weight                      → emb            (solo para verificar; luego se trasplanta)
  model.layers.i.input_layernorm.weight          → pre.i.g1
  model.layers.i.self_attn.qkv_proj.weight       → pre.i.wq, wk, wv   (K y V de 8 cabezas se repiten a 24: exacto)
  model.layers.i.self_attn.o_proj.weight         → pre.i.wo
  model.layers.i.post_attention_layernorm.weight → pre.i.g2
  model.layers.i.mlp.gate_up_proj.weight         → pre.i.w1 (gate), pre.i.w3 (up)
  model.layers.i.mlp.down_proj.weight            → pre.i.w2
  model.norm.weight                              → norm_f
Escalas: res_scale = 1 y logit_scale_fixed = 1 (Phi no escala ramas ni logits); RoPE parcial 0,75 y
LongRoPE (factores cortos y mscale) desde config.json.

    python -m navros.porte_phi --hf /ruta/Phi-4-mini-instruct --salida phi4mini_navros32k.pt
    python -m navros.porte_phi --autotest          # equivalencia con una referencia pequeña, sin descargas
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import NavrosConfig
from .pt.model import Navros


# ----------------------------------------------------------------------------- configuración
def config_desde_hf(hf: dict, vocab: int | None = None) -> NavrosConfig:
    """NavrosConfig equivalente a un config.json de Phi3ForCausalLM (Phi-4-mini)."""
    assert hf.get("hidden_act", "silu") == "silu" and not hf.get("attention_bias", False), "solo SwiGLU sin sesgos"
    d, H, Hkv = hf["hidden_size"], hf["num_attention_heads"], hf.get("num_key_value_heads", hf["num_attention_heads"])
    assert H % Hkv == 0
    frac = hf.get("partial_rotary_factor", 1.0)
    rs = hf.get("rope_scaling") or {}
    factors, mscale = (), 1.0
    if rs:
        assert rs.get("type", rs.get("rope_type")) == "longrope", rs
        factors = tuple(float(f) for f in rs["short_factor"])   # T ≤ original_max_position_embeddings
        escala = hf["max_position_embeddings"] / hf["original_max_position_embeddings"]
        mscale = math.sqrt(1 + math.log(escala) / math.log(hf["original_max_position_embeddings"]))
    return NavrosConfig(vocab=vocab or hf["vocab_size"], d=d, n_heads=H, ffn=hf["intermediate_size"],
                        n_pre=hf["num_hidden_layers"], n_core=0, causal=True, rope=True,
                        rope_theta=float(hf.get("rope_theta", 10000.0)), rope_frac=frac, rope_factors=factors,
                        rope_mscale=mscale, res_scale=1.0, logit_scale_fixed=1.0, norm_eps=float(hf["rms_norm_eps"]))


# ----------------------------------------------------------------------------- pesos
def mapear(hf_state: dict, cfg: NavrosConfig, Hkv: int) -> dict:
    """state_dict de HF (Phi3ForCausalLM) → dict con los nombres de NAVROS, en float32."""
    d, H, hd = cfg.d, cfg.n_heads, cfg.head_dim
    rep = H // Hkv
    out = {"emb": hf_state["model.embed_tokens.weight"].float(), "norm_f": hf_state["model.norm.weight"].float()}
    for i in range(cfg.n_pre):
        p = f"model.layers.{i}."
        qkv = hf_state[p + "self_attn.qkv_proj.weight"].float()          # (d + 2·Hkv·hd, d)
        q, k, v = qkv[:d], qkv[d:d + Hkv * hd], qkv[d + Hkv * hd:]
        expande = lambda w: w.view(Hkv, hd, d).repeat_interleave(rep, dim=0).reshape(H * hd, d)
        gu = hf_state[p + "mlp.gate_up_proj.weight"].float()             # (2·ffn, d): gate, up
        out.update({
            f"pre.{i}.g1": hf_state[p + "input_layernorm.weight"].float(),
            f"pre.{i}.wq": q, f"pre.{i}.wk": expande(k), f"pre.{i}.wv": expande(v),
            f"pre.{i}.wo": hf_state[p + "self_attn.o_proj.weight"].float(),
            f"pre.{i}.g2": hf_state[p + "post_attention_layernorm.weight"].float(),
            f"pre.{i}.w1": gu[:cfg.ffn], f"pre.{i}.w3": gu[cfg.ffn:],
            f"pre.{i}.w2": hf_state[p + "mlp.down_proj.weight"].float(),
        })
    return out


@torch.no_grad()
def trasplantar_embedding(emb_phi: torch.Tensor, tok_phi, tok_navros, eot_phi: int) -> torch.Tensor:
    """Embedding de 32K a partir de la de Phi: cada token nuestro es la media de las embeddings de Phi
    de los tokens en que Phi parte su texto. eot (id 0) copia el <|endoftext|> de Phi; pad (id 1) es cero."""
    V = tok_navros.get_vocab_size()
    nueva = torch.zeros(V, emb_phi.shape[1], dtype=torch.float32)
    sin_texto = 0
    for i in range(V):
        texto = tok_navros.decode([i])
        if i == 0 or not texto:
            nueva[i] = emb_phi[eot_phi].float()
            sin_texto += i != 0
            continue
        if i == 1:
            continue
        ids = tok_phi.encode(texto, add_special_tokens=False)
        ids = ids.ids if hasattr(ids, "ids") else ids
        nueva[i] = emb_phi[ids].float().mean(0) if len(ids) else emb_phi[eot_phi].float()
    return nueva, sin_texto


def exportar(state: dict, cfg: NavrosConfig, salida: str, extra: dict | None = None) -> dict:
    """Export en el formato de cloud/modal_pesos.py (init_from de navros/lm_ddp.py)."""
    modelo = Navros(cfg)
    faltan = set(dict(modelo.named_parameters())) ^ set(state)
    assert not faltan, f"claves que no cuadran: {sorted(faltan)[:5]}"
    for n, p in modelo.named_parameters():
        assert p.shape == state[n].shape, (n, p.shape, state[n].shape)
    meta = dict(step=0, preset="phi4mini", tag="porte-phi4mini", vocab=cfg.vocab, cfg=cfg.to_dict(), **(extra or {}))
    torch.save(dict(meta, state={n: t.to(torch.bfloat16) for n, t in state.items()}), salida)
    return meta


# ----------------------------------------------------------------------------- referencia (autotest)
class _ReferenciaPhi3(torch.nn.Module):
    """Implementación mínima e independiente de Phi3ForCausalLM (GQA, qkv y gate_up fusionados, RoPE parcial
    con LongRoPE) para comprobar el mapeo sin descargar nada."""

    def __init__(self, hf: dict):
        super().__init__()
        self.hf = hf
        d, H, Hkv, f, L, V = (hf[k] for k in ("hidden_size", "num_attention_heads", "num_key_value_heads",
                                               "intermediate_size", "num_hidden_layers", "vocab_size"))
        self.hd = d // H
        g = torch.Generator().manual_seed(1)
        rnd = lambda *s, esc=0.2: torch.randn(*s, generator=g) * esc
        st = {"model.embed_tokens.weight": rnd(V, d, esc=1.0), "model.norm.weight": 1 + rnd(d)}
        for i in range(L):
            p = f"model.layers.{i}."
            st[p + "input_layernorm.weight"] = 1 + rnd(d)
            st[p + "post_attention_layernorm.weight"] = 1 + rnd(d)
            st[p + "self_attn.qkv_proj.weight"] = rnd(d + 2 * Hkv * self.hd, d)
            st[p + "self_attn.o_proj.weight"] = rnd(d, d)
            st[p + "mlp.gate_up_proj.weight"] = rnd(2 * f, d)
            st[p + "mlp.down_proj.weight"] = rnd(d, f)
        self.st = st

    def forward(self, tokens):
        hf, st = self.hf, self.st
        d, H, Hkv, f, L = (hf[k] for k in ("hidden_size", "num_attention_heads", "num_key_value_heads",
                                            "intermediate_size", "num_hidden_layers"))
        hd, B, T = self.hd, *tokens.shape
        rd = int(hd * hf["partial_rotary_factor"])
        rs = hf["rope_scaling"]
        inv = 1.0 / (torch.tensor(rs["short_factor"], dtype=torch.float64) * hf["rope_theta"] ** (torch.arange(0, rd, 2, dtype=torch.float64) / rd))
        escala = hf["max_position_embeddings"] / hf["original_max_position_embeddings"]
        mscale = math.sqrt(1 + math.log(escala) / math.log(hf["original_max_position_embeddings"]))
        ang = torch.outer(torch.arange(T, dtype=torch.float64), inv)
        emb = torch.cat([ang, ang], -1)
        cos, sin = (emb.cos() * mscale).float(), (emb.sin() * mscale).float()

        def rot(x):  # x: (B, h, T, hd); HF: rotate_half sobre las primeras rd dims
            xr, xp = x[..., :rd], x[..., rd:]
            x1, x2 = xr[..., : rd // 2], xr[..., rd // 2:]
            rh = torch.cat([-x2, x1], -1)
            return torch.cat([xr * cos + rh * sin, xp], -1)

        def rms(x, w):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + hf["rms_norm_eps"]) * w

        x = st["model.embed_tokens.weight"][tokens]
        for i in range(L):
            p = f"model.layers.{i}."
            u = rms(x, st[p + "input_layernorm.weight"])
            qkv = u @ st[p + "self_attn.qkv_proj.weight"].T
            q, k, v = qkv[..., :d], qkv[..., d:d + Hkv * hd], qkv[..., d + Hkv * hd:]
            q = q.view(B, T, H, hd).transpose(1, 2)
            k = k.view(B, T, Hkv, hd).transpose(1, 2).repeat_interleave(H // Hkv, dim=1)
            v = v.view(B, T, Hkv, hd).transpose(1, 2).repeat_interleave(H // Hkv, dim=1)
            q, k = rot(q), rot(k)
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            x = x + o.transpose(1, 2).reshape(B, T, d) @ st[p + "self_attn.o_proj.weight"].T
            u = rms(x, st[p + "post_attention_layernorm.weight"])
            gu = u @ st[p + "mlp.gate_up_proj.weight"].T
            x = x + (F.silu(gu[..., :f]) * gu[..., f:]) @ st[p + "mlp.down_proj.weight"].T
        return rms(x, st["model.norm.weight"]) @ st["model.embed_tokens.weight"].T


def autotest():
    hf = dict(hidden_size=48, num_attention_heads=6, num_key_value_heads=2, intermediate_size=64, num_hidden_layers=3,
              vocab_size=100, partial_rotary_factor=0.75, rope_theta=10000.0, rms_norm_eps=1e-5, hidden_act="silu",
              attention_bias=False, tie_word_embeddings=True, max_position_embeddings=131072,
              original_max_position_embeddings=4096,
              rope_scaling=dict(type="longrope", short_factor=[1.0, 1.1, 1.0], long_factor=[1.0, 1.0, 1.0]))
    ref = _ReferenciaPhi3(hf)
    cfg = config_desde_hf(hf)
    modelo = Navros(cfg)
    state = mapear(ref.st, cfg, hf["num_key_value_heads"])
    with torch.no_grad():
        for n, p in modelo.named_parameters():
            p.copy_(state[n])
    tokens = torch.randint(0, hf["vocab_size"], (2, 11))
    with torch.no_grad():
        a, b = ref(tokens), modelo(tokens)
    err = (a - b).abs().max().item()
    print(f"autotest: rotary_dim={cfg.rotary_dim} mscale={cfg.rope_mscale:.4f} factores={cfg.rope_factors} "
          f"| error máximo entre referencia Phi3 y NAVROS portado: {err:.2e} (logits de magnitud {a.abs().max().item():.1f})")
    assert err < 1e-4, "el porte no es equivalente"
    print("autotest ok")


# ----------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--autotest", action="store_true")
    ap.add_argument("--hf", help="carpeta con config.json, model*.safetensors y tokenizer.json de Phi-4-mini")
    ap.add_argument("--tokenizer", default=str(Path(__file__).resolve().parent / "assets" / "tokenizer_navros_32k.json"))
    ap.add_argument("--salida", default="phi4mini_navros32k.pt")
    ap.add_argument("--verificar", type=int, default=2, help="secuencias para comparar con transformers (0 = no)")
    a = ap.parse_args()
    if a.autotest:
        autotest()
        return
    from safetensors.torch import load_file
    from tokenizers import Tokenizer
    hf = json.loads((Path(a.hf) / "config.json").read_text())
    assert "Phi3ForCausalLM" in hf.get("architectures", []), hf.get("architectures")
    assert hf.get("tie_word_embeddings", False), "el porte asume embedding compartida (Phi-4-mini la tiene)"
    st = {}
    for f in sorted(Path(a.hf).glob("*.safetensors")):
        st.update(load_file(str(f)))
    cfg_phi = config_desde_hf(hf)
    state = mapear(st, cfg_phi, hf["num_key_value_heads"])
    print(f"cuerpo mapeado: {cfg_phi.n_pre} capas, d={cfg_phi.d}, rotary {cfg_phi.rotary_dim}, mscale {cfg_phi.rope_mscale:.4f}")

    if a.verificar:  # con la embedding ORIGINAL de Phi: la prueba de que el cuerpo es el mismo
        from transformers import AutoModelForCausalLM
        modelo = Navros(cfg_phi)
        with torch.no_grad():
            for n, p in modelo.named_parameters():
                p.copy_(state[n])
        ref = AutoModelForCausalLM.from_pretrained(a.hf, torch_dtype=torch.float32)
        tokens = torch.randint(0, hf["vocab_size"], (a.verificar, 64))
        with torch.no_grad():
            la, lb = ref(tokens).logits, modelo(tokens)
        err = (la - lb).abs().max().item()
        print(f"verificación contra transformers: error máximo {err:.3e} sobre logits de magnitud {la.abs().max().item():.1f}")
        assert err < 1e-2, "el cuerpo portado no coincide con transformers"
        del ref

    tok_phi = Tokenizer.from_file(str(Path(a.hf) / "tokenizer.json"))
    tok_nav = Tokenizer.from_file(a.tokenizer)
    eot_phi = hf.get("eos_token_id", 199999)
    eot_phi = eot_phi[0] if isinstance(eot_phi, list) else eot_phi
    nueva, sin_texto = trasplantar_embedding(state["emb"], tok_phi, tok_nav, eot_phi)
    print(f"embedding trasplantada: {tuple(nueva.shape)}, {sin_texto} tokens sin texto (copian eot); "
          f"norma media Phi {state['emb'].float().norm(dim=1).mean():.4f} → nueva {nueva.norm(dim=1).mean():.4f}")
    cfg = config_desde_hf(hf, vocab=tok_nav.get_vocab_size())
    state["emb"] = nueva
    meta = exportar(state, cfg, a.salida, extra=dict(origen=hf.get("_name_or_path", a.hf), licencia_origen="MIT",
                                                      tokenizador="navros_32k", trasplante="media de subtokens"))
    print("EXPORT", json.dumps({k: v for k, v in meta.items() if k != "cfg"}))


if __name__ == "__main__":
    main()
