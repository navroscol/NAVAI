"""Port PyTorch de NAVROS, verificado contra el oráculo NumPy (scripts/02_verify_port.py)."""
from .model import Navros, weighted_xent
from .optim import Muon, AdamW, newton_schulz5, clip_grad_norm

__all__ = ["Navros", "weighted_xent", "Muon", "AdamW", "newton_schulz5", "clip_grad_norm"]
