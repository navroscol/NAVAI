"""Oráculo NumPy: backward manual, sin dependencias salvo NumPy. Corre en Windows sin CUDA."""
from .model import NavrosNP, init_params, sample_r
from .optim import MuonNP, AdamWNP, newton_schulz5, clip_grads

__all__ = ["NavrosNP", "init_params", "sample_r", "MuonNP", "AdamWNP", "newton_schulz5", "clip_grads"]
