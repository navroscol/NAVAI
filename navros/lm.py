"""NAVROS-LM: entrenamiento del modelo de lenguaje sobre el corpus pretokenizado.

- Datos deterministas y reanudables: cada idioma recorre una permutación fija de trozos de
  T+1 tokens; el estado del lector es un cursor por idioma que va en el checkpoint.
- Mezcla de idiomas fija por lote (por defecto 50/50).
- Recurrente: r ~ Poisson-lognormal por paso (el mismo r en todos los micro-lotes del paso),
  BPTT truncado por las últimas k iteraciones. El r de cada paso sale de un RNG sembrado
  con (semilla, paso): al reanudar se repite exactamente la misma secuencia de r.
- Checkpoint cada N minutos y al agotar el presupuesto de tiempo; al arrancar se reanuda
  desde el checkpoint más avanzado que encuentre (propio o adjunto como entrada).
- Evaluación: pérdida de validación por idioma a r̄ y, para el recurrente, la curva de
  pérdida en función de r (¿pensar más ayuda?) y la salida adaptativa por token.
"""
from __future__ import annotations

import glob
import json
import math
import os
import time
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import NavrosConfig
from .oracle.model import sample_r
from .pt.model import Navros
from .pt.optim import Muon, clip_grad_norm


# ----------------------------------------------------------------------- arquitecturas
def lm_preset(name: str, vocab: int = 32768) -> NavrosConfig:
    P = {
        # estudio pequeño (T4): recurrente contra pilas fijas de mismo cómputo y mismos parámetros
        "rec-s":    dict(d=512, n_heads=8, n_pre=2, n_core=4, n_blocks=1, n_coda=2, r_mean=4, k_bptt=2, r_max=16),
        "fix20-s":  dict(d=512, n_heads=8, n_pre=20, n_core=0),       # mismo cómputo de forward que rec-s a r̄=4
        "fix8-s":   dict(d=512, n_heads=8, n_pre=8, n_core=0),        # mismos parámetros únicos que rec-s
        # NAVROS-1B (TPU v5e-8): 18 capas únicas, núcleo de 8 iterado
        "navros-1b": dict(d=2048, n_heads=16, ffn=6144, n_pre=4, n_core=8, n_blocks=1, n_coda=6, r_mean=4, k_bptt=2, r_max=16),
        "navros-1b-fix": dict(d=2048, n_heads=16, ffn=6144, n_pre=18, n_core=0),
        # humo
        "tiny":     dict(d=64, n_heads=2, n_pre=1, n_core=1, n_blocks=1, n_coda=1, r_mean=3, k_bptt=2, r_max=6),
    }[name]
    return NavrosConfig(vocab=vocab, causal=True, rope=True, **P)


def train_flops_per_token(cfg: NavrosConfig, r: float, k: float, T: int) -> float:
    """Aproximación estándar: 2·N por token y capa en forward, 4·N en backward, más atención."""
    d, f = cfg.d, cfg.ffn
    layer = 2 * (4 * d * d + 3 * d * f) + 4 * T * d / 2  # atención causal: la mitad de la matriz T×T
    fwd = cfg.n_pre + cfg.n_coda + cfg.n_core * r
    bwd = cfg.n_pre + cfg.n_coda + cfg.n_core * k
    return fwd * layer + 2 * bwd * layer + 6 * cfg.vocab * d


# ------------------------------------------------------------------------------ datos
def find_data_dirs(pattern="/kaggle/input/**/navros_data/manifest.json"):
    return sorted({str(Path(p).parent) for p in glob.glob(pattern, recursive=True)})


class TokenData:
    def __init__(self, dirs, langs: dict, T: int, seed: int = 0):
        self.T, self.langs = T, langs
        self.manifests = [json.loads((Path(d) / "manifest.json").read_text()) for d in dirs]
        self.dirs = [Path(d) for d in dirs]
        self.train, self.perm, self.cum, self.cursor = {}, {}, {}, {}
        for lang in langs:
            maps = []
            for d, m in zip(self.dirs, self.manifests):
                for s in m["langs"][lang]["train"]["shards"]:
                    maps.append(np.memmap(d / s["file"], dtype=np.uint16, mode="r"))
            counts = np.array([len(a) // (T + 1) for a in maps], dtype=np.int64)
            self.train[lang] = maps
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
        return self.train[lang][s][off:off + self.T + 1]

    def batch(self, B):
        per = self._split(B)
        rows = []
        for lang, n in per.items():
            for _ in range(n):
                c = self.cursor[lang]
                rows.append(self._chunk(lang, self.perm[lang][c % len(self.perm[lang])]))
                self.cursor[lang] = c + 1
        x = torch.from_numpy(np.stack(rows).astype(np.int64))
        return x[:, :-1], x[:, 1:]

    def _split(self, B):
        langs = list(self.langs)
        per = {l: int(round(B * self.langs[l])) for l in langs}
        per[langs[-1]] = B - sum(per[l] for l in langs[:-1])
        return per

    def eval_set(self, lang, split, n_seq):
        """Primeros n_seq trozos contiguos de val/test (deterministas)."""
        d, m = self.dirs[0], self.manifests[0]
        arr = np.memmap(d / m["langs"][lang][split]["file"], dtype=np.uint16, mode="r")
        n = min(n_seq, len(arr) // (self.T + 1))
        x = torch.from_numpy(np.asarray(arr[:n * (self.T + 1)]).reshape(n, self.T + 1).astype(np.int64))
        return x[:, :-1], x[:, 1:]

    def state(self):
        return dict(cursor=dict(self.cursor))

    def load_state(self, st):
        self.cursor.update(st["cursor"])


class SyntheticData:
    """Tokens aleatorios con la misma interfaz que TokenData: para medir velocidad y memoria
    sin corpus. Determinista por cursor (reanudable)."""

    def __init__(self, vocab, langs, T, seed=0):
        self.vocab, self.langs, self.T, self.seed = vocab, langs, T, seed
        self.cursor = {l: 0 for l in langs}

    def batch(self, B):
        c = self.cursor[next(iter(self.langs))]
        x = np.random.default_rng([self.seed, c]).integers(0, self.vocab, (B, self.T + 1))
        for l in self.langs:
            self.cursor[l] += B
        x = torch.from_numpy(x.astype(np.int64))
        return x[:, :-1], x[:, 1:]

    def eval_set(self, lang, split, n_seq):
        x = np.random.default_rng([self.seed, 999, zlib.crc32((lang + split).encode())]).integers(0, self.vocab, (n_seq, self.T + 1))
        x = torch.from_numpy(x.astype(np.int64))
        return x[:, :-1], x[:, 1:]

    def state(self):
        return dict(cursor=dict(self.cursor))

    def load_state(self, st):
        self.cursor.update(st["cursor"])


def make_data(rc):
    if rc.synthetic:
        return SyntheticData(32768, rc.langs, rc.T, rc.seed)
    return TokenData(rc.data_dirs or find_data_dirs(), rc.langs, rc.T, rc.seed)


# ---------------------------------------------------------------------- configuración
@dataclass
class LMRun:
    preset: str = "tiny"
    data_dirs: list = field(default_factory=list)
    langs: dict = field(default_factory=lambda: {"es": 0.5, "en": 0.5})
    T: int = 1024
    batch: int = 64                 # secuencias por paso
    micro: int = 8                  # secuencias por micro-lote
    tokens: int = 200_000_000       # tokens totales de entrenamiento
    lr_muon: float = 0.02
    lr_adam: float = 3e-3
    wd: float = 0.0
    warmup: int = 200               # pasos
    decay_frac: float = 0.2         # WSD: decaimiento lineal a 0 en el último 20 %
    clip: float = 1.0
    precision: str = "fp16"         # fp16 (T4) | bf16 | fp32
    seed: int = 0
    device: str = "cuda"
    eval_every: int = 500
    eval_seq: int = 64              # secuencias de validación por idioma
    r_eval: tuple = (1, 2, 3, 4, 6, 8, 12, 16)
    adaptive_tol: float = 0.01
    ckpt_dir: str = ""
    ckpt_every_min: float = 30.0
    grad_ckpt: bool = False         # recomputar activaciones por capa en el backward
    xla_cache: str = ""             # caché persistente de grafos compilados (TPU)
    synthetic: bool = False         # tokens aleatorios (benchmark de velocidad/memoria)
    spmd: bool = True               # XLA: repartir entre chips (False para un solo chip, p. ej. v5e-1)
    muon_buf: str = "fp32"          # precisión del momento de Muon: fp32 | bf16
    time_budget_s: float = 1e12
    log_every: int = 50
    tag: str = ""

    @property
    def steps(self):
        return max(1, self.tokens // (self.batch * self.T))


def lr_mult(step, rc: LMRun):
    if step < rc.warmup:
        return (step + 1) / rc.warmup
    d0 = int(rc.steps * (1 - rc.decay_frac))
    if step < d0:
        return 1.0
    return max(0.0, (rc.steps - step) / max(1, rc.steps - d0))


# ------------------------------------------------------------------------ evaluación
@torch.no_grad()
def eval_losses(model, xy, rc: LMRun, be, r=None):
    model.eval()
    x, y = xy
    tot, n = 0.0, 0
    for i in range(0, len(x) - len(x) % rc.micro or len(x), rc.micro):
        xb, yb = be.shard_batch(x[i:i + rc.micro].to(be.device)), be.shard_batch(y[i:i + rc.micro].to(be.device))
        with be.autocast():
            logits = model(xb, r=r)
        tot += float(F.cross_entropy(logits.float().flatten(0, 1), yb.flatten(), reduction="sum"))
        n += yb.numel()
    model.train()
    return tot / n


@torch.no_grad()
def eval_recurrence(model, xy, rc: LMRun, be):
    """Pérdida tras cada iteración (curva) y con salida adaptativa por token: la posición se
    congela cuando ‖Δh‖/‖h‖ < tol; las iteraciones siguientes ven su estado congelado."""
    model.eval()
    cfg = model.cfg
    rmax = max(rc.r_eval)
    x, y = xy
    curve = np.zeros(rmax)
    ada_loss, ada_r, n = 0.0, 0.0, 0
    device = be.device
    for i in range(0, len(x) - len(x) % rc.micro or len(x), rc.micro):
        xb, yb = be.shard_batch(x[i:i + rc.micro].to(device)), be.shard_batch(y[i:i + rc.micro].to(device))
        with be.autocast():
            x0, rope, mask = model.prelude(xb)
            h = torch.zeros_like(x0)
            done = torch.zeros(xb.shape, dtype=torch.bool, device=device)
            used = torch.full(xb.shape, rmax, device=device)
            ada_nll = None
            for t in range(rmax):
                hn = model._iter(h, x0, 0, rope, mask)
                delta = (hn - h).float().norm(dim=-1) / hn.float().norm(dim=-1).clamp_min(1e-6)
                hn = torch.where(done[..., None], h, hn)
                h = hn
                logits = model.coda_logits(h, rope, mask).float()
                nll = F.cross_entropy(logits.flatten(0, 1), yb.flatten(), reduction="none").view_as(yb)
                curve[t] += float(nll.sum())
                newly = (~done) & (delta < rc.adaptive_tol)
                used = torch.where(newly, torch.full_like(used, t + 1), used)
                ada_nll = nll if ada_nll is None else torch.where(done, ada_nll, nll)
                done |= newly
                be.barrier()
        ada_loss += float(ada_nll.sum())
        ada_r += float(used.sum())
        n += yb.numel()
    model.train()
    return dict(curve={int(r): curve[r - 1] / n for r in rc.r_eval}, adaptive=dict(loss=ada_loss / n, mean_r=ada_r / n))


# --------------------------------------------------------------------- checkpoints
def latest_checkpoint(rc: LMRun):
    cands = []
    for pat in ([f"{rc.ckpt_dir}/latest.pt"] if rc.ckpt_dir else []) + ["/kaggle/input/**/ckpt/latest.pt"]:
        for p in glob.glob(pat, recursive=True):
            try:
                meta = json.loads(Path(p).with_suffix(".json").read_text())
                if meta.get("preset") == rc.preset and meta.get("tag") == rc.tag:
                    cands.append((meta["step"], p))
            except Exception:
                pass
    return max(cands)[1] if cands else None


def save_checkpoint(rc, step, model, opt, data, scaler, extra, be):
    if not rc.ckpt_dir:
        return
    d = Path(rc.ckpt_dir)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "latest.tmp"
    be.save(dict(step=step, model=model.state_dict(), opt=opt.state_dict(), opt_t=opt.t, data=data.state(),
                 scaler=scaler.state_dict() if scaler is not None else None, extra=extra), tmp)
    os.replace(tmp, d / "latest.pt")
    (d / "latest.json").write_text(json.dumps(dict(step=step, preset=rc.preset, tag=rc.tag, time=time.time())))


R_BUCKETS = (1, 2, 3, 4, 5, 6, 8, 10, 12, 16)


def step_r(rc: LMRun, cfg: NavrosConfig, step: int, be):
    """r y k del paso. En XLA r se redondea al cubo más cercano de R_BUCKETS: cada (r, k)
    distinto es un grafo compilado distinto, así que se limita su número."""
    if not cfg.recurrent:
        return None, None
    r = sample_r(np.random.default_rng([rc.seed, step, 7]), cfg)
    if be.is_xla:
        r = min(R_BUCKETS, key=lambda b: (abs(b - r), b))
    return r, min(cfg.k_bptt, r)


# ------------------------------------------------------------------------ entrenamiento
def train(rc: LMRun, log=print) -> dict:
    t_start = time.time()
    from .backend import Backend
    be = Backend(rc.device, rc.precision, xla_cache=rc.xla_cache or None, spmd=rc.spmd)
    data = make_data(rc)
    cfg = lm_preset(rc.preset, data.vocab)
    torch.manual_seed(rc.seed)
    model = Navros(cfg).to(be.device)
    for p in model.parameters():
        be.shard_param(p)
    if rc.grad_ckpt:
        model.ckpt_fn = be.ckpt_fn()
    named = list(model.named_parameters())
    opt = Muon(named, lr_muon=rc.lr_muon, lr_adam=rc.lr_adam, wd_muon=rc.wd, tensor_scalars=be.is_xla,
               state_hook=be.shard_like, ns_dtype=torch.bfloat16 if be.is_xla else None,
               buf_dtype=torch.bfloat16 if rc.muon_buf == "bf16" else None)
    scaler = torch.amp.GradScaler("cuda") if (rc.precision == "fp16" and be.device.type == "cuda") else None
    evals = {l: data.eval_set(l, "val", rc.eval_seq) for l in rc.langs}
    step, history, extra = 0, [], {}
    ck = latest_checkpoint(rc)
    if ck:
        st = torch.load(ck, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        for p in model.parameters():
            for t in opt.state[p].values():
                if torch.is_tensor(t):
                    be.shard_like(t, p)
        opt.t = st["opt_t"]
        data.load_state(st["data"])
        if scaler is not None and st["scaler"]:
            scaler.load_state_dict(st["scaler"])
        step, extra = st["step"], st["extra"]
        history = extra.get("history", [])
        log(f"reanudado desde {ck} en el paso {step}")
    n_micro = rc.batch // rc.micro
    r_bar = round(cfg.r_mean) if cfg.recurrent else None
    flops_tok = train_flops_per_token(cfg, cfg.r_mean if cfg.recurrent else 0, min(cfg.k_bptt, cfg.r_mean) if cfg.recurrent else 0, rc.T)
    last_ck, t_log, tok_log = time.time(), time.time(), 0
    run = torch.zeros((), device=be.device)
    log(f"{rc.preset}: {sum(p.numel() for p in model.parameters()):,} params, {rc.steps} pasos, "
        f"{flops_tok/1e9:.2f} GFLOP/token de entrenamiento, dispositivo {rc.device} ×{be.n_dev}")
    while step < rc.steps:
        if time.time() - t_start > rc.time_budget_s:
            log("presupuesto de tiempo agotado: checkpoint y salida")
            break
        r, k = step_r(rc, cfg, step, be)
        opt.zero_grad(set_to_none=True)
        for _ in range(n_micro):
            x, y = data.batch(rc.micro)
            x = be.shard_batch(x.to(be.device, non_blocking=True))
            y = be.shard_batch(y.to(be.device, non_blocking=True))
            with be.autocast():
                logits = model(x, r=r, k=k)
            loss = F.cross_entropy(logits.float().flatten(0, 1), y.flatten()) / n_micro
            (scaler.scale(loss) if scaler else loss).backward()
            run = run + loss.detach()
            be.barrier()
        if scaler:
            scaler.unscale_(opt)
        gn = clip_grad_norm(model.parameters(), rc.clip)
        if scaler:
            scaler.step(opt, lr_mult(step, rc))
            scaler.update()
        else:
            opt.step(lr_mult(step, rc))
        be.barrier()
        step += 1
        tok_log += rc.batch * rc.T
        if step % rc.log_every == 0:
            dt = time.time() - t_log
            loss_avg = float(run) / rc.log_every
            row = dict(step=step, loss=loss_avg, gnorm=float(gn), tok_s=tok_log / dt, tflops=tok_log * flops_tok / dt / 1e12,
                       lr=lr_mult(step, rc), elapsed=time.time() - t_start)
            history.append(row)
            log(f"[{rc.tag or rc.preset}] paso {step}/{rc.steps} loss {loss_avg:.4f} |g| {row['gnorm']:.2f} "
                f"{row['tok_s']/1e3:.1f}K tok/s {row['tflops']:.1f} TFLOP/s")
            t_log, tok_log = time.time(), 0
            run = torch.zeros((), device=be.device)
            if not math.isfinite(loss_avg) and scaler is None:
                log(f"pérdida no finita en el paso {step}: se detiene")
                break
        if step % rc.eval_every == 0 or step == rc.steps:
            ev = {l: eval_losses(model, evals[l], rc, be, r=r_bar) for l in rc.langs}
            history.append(dict(step=step, val=ev))
            log(f"[{rc.tag or rc.preset}] paso {step} val " + " ".join(f"{l} {v:.4f}" for l, v in ev.items()))
        if time.time() - last_ck > rc.ckpt_every_min * 60:
            save_checkpoint(rc, step, model, opt, data, scaler, dict(history=history), be)
            last_ck = time.time()
    extra["history"] = history
    save_checkpoint(rc, step, model, opt, data, scaler, extra, be)
    done = step >= rc.steps
    res = dict(run=asdict(rc), model=cfg.to_dict(), params=sum(p.numel() for p in model.parameters()),
               flops_per_token=flops_tok, step=step, finished=done, history=history, seconds=time.time() - t_start)
    if done:
        res["val"] = {l: eval_losses(model, evals[l], rc, be, r=r_bar) for l in rc.langs}
        tests = {l: data.eval_set(l, "test", rc.eval_seq) for l in rc.langs}
        res["test"] = {l: eval_losses(model, tests[l], rc, be, r=r_bar) for l in rc.langs}
        if cfg.recurrent:
            res["recurrence"] = {l: eval_recurrence(model, evals[l], rc, be) for l in rc.langs}
    return res


def train_lm_run(kw):
    """Punto de entrada para experiments.run_grid."""
    rc = LMRun(**kw)
    return train(rc, log=lambda s: print(s, flush=True))


def lm_brief(res):
    if not res.get("finished"):
        return f"sin terminar (paso {res['step']})"
    return "test " + " ".join(f"{l} {v:.4f}" for l, v in res["test"].items())


def benchmark(preset, device="cpu", precision="fp32", T=1024, micro=8, n_micro=1, steps=10, warmup=3, r=None,
              grad_ckpt=False, vocab=32768, spmd=True, muon_buf="fp32", log=print):
    """Velocidad y memoria con tokens sintéticos (no necesita corpus). r fijo (por defecto r̄)
    para medir el régimen estable; el primer paso incluye la compilación en XLA."""
    from .backend import Backend
    be = Backend(device, precision, spmd=spmd)
    cfg = lm_preset(preset, vocab)
    torch.manual_seed(0)
    model = Navros(cfg).to(be.device)
    for p in model.parameters():
        be.shard_param(p)
    if grad_ckpt:
        model.ckpt_fn = be.ckpt_fn()
    opt = Muon(list(model.named_parameters()), tensor_scalars=be.is_xla, state_hook=be.shard_like,
               ns_dtype=torch.bfloat16 if be.is_xla else None,
               buf_dtype=torch.bfloat16 if muon_buf == "bf16" else None)
    if cfg.recurrent:
        r = r or round(cfg.r_mean)
        k = min(cfg.k_bptt, r)
    else:
        r = k = None
    flops = train_flops_per_token(cfg, r or 0, k or 0, T)
    g = torch.Generator().manual_seed(0)
    times, loss = [], None
    for step in range(warmup + steps):
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        for _ in range(n_micro):
            xy = torch.randint(0, vocab, (micro, T + 1), generator=g)
            x = be.shard_batch(xy[:, :-1].to(be.device))
            y = be.shard_batch(xy[:, 1:].to(be.device))
            with be.autocast():
                logits = model(x, r=r, k=k)
            loss = F.cross_entropy(logits.float().flatten(0, 1), y.flatten()) / n_micro
            loss.backward()
            be.barrier()
        clip_grad_norm(model.parameters(), 1.0)
        opt.step(1.0)
        be.barrier()
        lv = float(loss.detach())  # sincroniza
        times.append(time.time() - t0)
        log(f"  {preset} paso {step}: {times[-1]:.2f}s loss {lv:.3f}")
    steady = float(np.median(times[warmup:]))
    tok = micro * n_micro * T
    res = dict(preset=preset, params=sum(p.numel() for p in model.parameters()), T=T, micro=micro, n_micro=n_micro,
               r=r, k=k, grad_ckpt=grad_ckpt, muon_buf=muon_buf, first_step_s=times[0], step_s=steady, tok_s=tok / steady,
               tflops=tok * flops / steady / 1e12, flops_per_token=flops, loss=lv, n_dev=be.n_dev)
    if be.is_xla:
        try:
            res["mem"] = {k2: v for k2, v in be.xm.get_memory_info(be.device).items()}
        except Exception as e:
            res["mem"] = repr(e)[:200]
    elif be.device.type == "cuda":
        res["mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
    log(json.dumps(res, default=str))
    return res
