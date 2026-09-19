"""Muon + AdamW en PyTorch. Réplica operación a operación de navros/oracle/optim.py."""
from __future__ import annotations

import torch

NS_COEFS = (3.4445, -4.7750, 2.0315)


def is_muon_param(name: str, p: torch.Tensor) -> bool:
    return p.ndim == 2 and name not in ("emb", "abaco")


def newton_schulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7, dtype=None) -> torch.Tensor:
    """dtype=None conserva la precisión de G (verificación); en GPU/TPU se usa bfloat16."""
    a, b, c = NS_COEFS
    X = G if dtype is None else G.to(dtype)
    tr = G.shape[0] > G.shape[1]
    if tr:
        X = X.T
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if tr:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Híbrido: Muon para matrices ocultas, AdamW para emb/abaco/ganancias.

    params: iterable de (nombre, tensor), p. ej. model.named_parameters().
    step(lr_mult) escala ambos learning rates (para el schedule).

    tensor_scalars=True (XLA/TPU): el LR y las correcciones de sesgo, que cambian en cada
    paso, se pasan como tensores del dispositivo. Con floats de Python quedarían incrustados
    como constantes y XLA recompilaría el grafo del optimizador en cada paso.
    state_hook(estado, parámetro): se llama al crear cada tensor de estado (p. ej. para
    repartirlo entre chips igual que su parámetro).
    """

    def __init__(self, named_params, lr_muon=0.02, lr_adam=3e-3, momentum=0.95, nesterov=True,
                 ns_steps=5, wd_muon=0.0, wd_adam=0.0, betas=(0.9, 0.95), eps=1e-8,
                 use_muon=True, ns_dtype=None, tensor_scalars=False, state_hook=None, buf_dtype=None):
        named = [(n, p) for n, p in named_params if p.requires_grad]
        muon = [p for n, p in named if use_muon and is_muon_param(n, p)]
        adam = [p for n, p in named if not (use_muon and is_muon_param(n, p))]
        groups = []
        if muon:
            groups.append(dict(params=muon, kind="muon", lr=lr_muon, wd=wd_muon))
        if adam:
            groups.append(dict(params=adam, kind="adam", lr=lr_adam, wd=wd_adam))
        super().__init__(groups, dict(momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                                      betas=betas, eps=eps, ns_dtype=ns_dtype))
        self.t = 0
        self.tensor_scalars, self.state_hook, self.buf_dtype = tensor_scalars, state_hook, buf_dtype

    def _new_state(self, p, dtype=None):
        z = torch.zeros_like(p, dtype=dtype)
        if self.state_hook is not None:
            self.state_hook(z, p)
        return z

    @torch.no_grad()
    def step(self, lr_mult: float = 1.0):
        if self.tensor_scalars:
            return self._step_tensor(lr_mult)
        self.t += 1
        for group in self.param_groups:
            lr = group["lr"] * lr_mult
            if group["kind"] == "muon":
                beta = group["momentum"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    st = self.state[p]
                    if "buf" not in st:
                        st["buf"] = self._new_state(p, self.buf_dtype)
                    buf = st["buf"]
                    buf.lerp_(g.to(buf.dtype), 1.0 - beta)
                    bg = buf.to(g.dtype)
                    u = torch.lerp(g, bg, beta) if group["nesterov"] else bg
                    O = newton_schulz5(u, group["ns_steps"], dtype=group["ns_dtype"])
                    O = O * max(1.0, u.shape[0] / u.shape[1]) ** 0.5
                    p.mul_(1.0 - lr * group["wd"])
                    p.add_(O, alpha=-lr)
            else:
                b1, b2 = group["betas"]
                bc1, bc2 = 1.0 - b1 ** self.t, 1.0 - b2 ** self.t
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    st = self.state[p]
                    if "m" not in st:
                        st["m"] = self._new_state(p)
                        st["v"] = self._new_state(p)
                    m, v = st["m"], st["v"]
                    m.lerp_(g, 1.0 - b1)
                    v.mul_(b2).addcmul_(g, g, value=1.0 - b2)
                    if p.ndim == 2:
                        p.mul_(1.0 - lr * group["wd"])
                    p.addcdiv_(m, v.sqrt() / bc2 ** 0.5 + group["eps"], value=-(lr / bc1))

    def _step_tensor(self, lr_mult):
        """Misma matemática que step(), con los escalares variables como tensores."""
        self.t += 1
        first = next(p for g in self.param_groups for p in g["params"])
        b1, b2 = self.defaults["betas"]
        vals = [g["lr"] * lr_mult for g in self.param_groups] + [1.0 - b1 ** self.t, 1.0 - b2 ** self.t]
        sc = torch.tensor(vals, dtype=torch.float32).to(first.device)  # transferencia de datos, no constante
        bc1, bc2 = sc[-2], sc[-1]
        for gi, group in enumerate(self.param_groups):
            lr = sc[gi]
            if group["kind"] == "muon":
                beta = group["momentum"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    st = self.state[p]
                    if "buf" not in st:
                        st["buf"] = self._new_state(p, self.buf_dtype)
                    buf = st["buf"]
                    buf.lerp_(g.to(buf.dtype), 1.0 - beta)
                    bg = buf.to(g.dtype)
                    u = torch.lerp(g, bg, beta) if group["nesterov"] else bg
                    O = newton_schulz5(u, group["ns_steps"], dtype=group["ns_dtype"])
                    O = O * max(1.0, u.shape[0] / u.shape[1]) ** 0.5
                    p.mul_((1.0 - lr * group["wd"]).to(p.dtype))
                    p.sub_(O * lr.to(p.dtype))
            else:
                eps = group["eps"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    st = self.state[p]
                    if "m" not in st:
                        st["m"] = self._new_state(p)
                        st["v"] = self._new_state(p)
                    m, v = st["m"], st["v"]
                    m.lerp_(g, 1.0 - b1)
                    v.mul_(b2).addcmul_(g, g, value=1.0 - b2)
                    if p.ndim == 2:
                        p.mul_((1.0 - lr * group["wd"]).to(p.dtype))
                    p.sub_((lr / bc1).to(p.dtype) * m / (v.sqrt() / bc2.sqrt().to(p.dtype) + eps))


def AdamW(named_params, lr=3e-3, wd=0.0, betas=(0.9, 0.95), eps=1e-8):
    return Muon(named_params, lr_adam=lr, wd_adam=wd, betas=betas, eps=eps, use_muon=False)


def clip_grad_norm(params, max_norm):
    params = [p for p in params if p.grad is not None]
    total = torch.sqrt(sum((p.grad.detach().float() ** 2).sum() for p in params))
    if max_norm:
        s = torch.clamp(max_norm / (total + 1e-6), max=1.0)
        for p in params:
            p.grad.mul_(s.to(p.grad.dtype))
    return total
