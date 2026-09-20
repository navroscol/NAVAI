"""Dispositivo de entrenamiento: CPU / CUDA / MPS / XLA (TPU), con la misma interfaz.

En XLA (TPU v5e-8) se usa SPMD en un solo proceso:
  * malla de todos los chips en un eje "fsdp";
  * cada matriz se reparte por filas entre chips (estilo FSDP); vectores replicados;
  * los lotes se reparten por la dimensión de batch (paralelismo de datos);
  * el estado del optimizador se reparte igual que su parámetro (state_hook de Muon);
  * xm.mark_step() corta el grafo tras cada micro-lote y tras cada paso del optimizador,
    así el mismo grafo compilado se reutiliza.
"""
from __future__ import annotations

import contextlib

import numpy as np
import torch


class Backend:
    def __init__(self, device: str, precision: str, xla_cache: str | None = None, spmd: bool = True):
        self.is_xla = device == "xla"
        self.dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]
        if self.is_xla:
            import torch_xla.core.xla_model as xm
            import torch_xla.runtime as xr
            self.spmd = spmd
            if spmd:
                xr.use_spmd()
            if xla_cache:
                try:
                    xr.initialize_cache(xla_cache, readonly=False)
                except Exception as e:  # la caché es una optimización, no un requisito
                    print("caché XLA no disponible:", repr(e)[:200], flush=True)
            self.xm = xm
            self.device = xm.xla_device()
            self.n_dev = xr.global_runtime_device_count() if spmd else 1
            if spmd:
                import torch_xla.distributed.spmd as xs
                self.xs = xs
                self.mesh = xs.Mesh(np.arange(self.n_dev), (self.n_dev, 1), ("fsdp", "model"))
            self.autocast_type = "xla"
        else:
            self.device = torch.device(device)
            self.n_dev = 1
            self.spmd = False
            self.autocast_type = self.device.type

    # --- reparto ---------------------------------------------------------------------
    def shard_param(self, p):
        if self.is_xla and self.spmd and p.ndim == 2 and p.shape[0] % self.n_dev == 0:
            self.xs.mark_sharding(p, self.mesh, ("fsdp", None))

    def shard_like(self, t, p):
        self.shard_param(t) if t.shape == p.shape else None

    def shard_batch(self, x):
        if self.is_xla and self.spmd:
            self.xs.mark_sharding(x, self.mesh, ("fsdp", None))
        return x

    # --- ejecución -------------------------------------------------------------------
    def autocast(self):
        if self.dtype == torch.float32:
            return contextlib.nullcontext()
        return torch.autocast(self.autocast_type, dtype=self.dtype)

    def barrier(self):
        if self.is_xla:
            self.xm.mark_step()

    def ckpt_fn(self):
        if self.is_xla:
            from torch_xla.utils.checkpoint import checkpoint as xla_ckpt
            return lambda f, *a: xla_ckpt(f, *a, use_reentrant=False)
        import torch.utils.checkpoint as ckpt
        return lambda f, *a: ckpt.checkpoint(f, *a, use_reentrant=False)

    def save(self, obj, path):
        if self.is_xla:
            self.xm.save(obj, str(path))  # reúne los tensores repartidos en la CPU
        else:
            torch.save(obj, path)
