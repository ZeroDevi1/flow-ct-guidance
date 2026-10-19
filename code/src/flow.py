"""流匹配（rectified flow / conditional flow matching）核心：训练目标与 ODE 采样。

约定：t=0 为噪声 ε~N(0,I)，t=1 为数据 x。
    x_t = (1 - t) ε + t x,     目标速度 v* = x - ε
    由 (x_t, v) 可得:  x1_hat = x_t + (1 - t) v,   ε_hat = x_t - t v
"""
from __future__ import annotations

import torch


def sample_t(batch: int, device, mode: str = "logit-normal") -> torch.Tensor:
    if mode == "uniform":
        return torch.rand(batch, device=device)
    # SD3 风格 logit-normal，多采中段时间
    return torch.sigmoid(torch.randn(batch, device=device))


def fm_loss(model, x: torch.Tensor, t_mode: str = "logit-normal") -> torch.Tensor:
    b = x.shape[0]
    t = sample_t(b, x.device, t_mode)
    eps = torch.randn_like(x)
    tt = t.view(-1, 1, 1, 1)
    xt = (1 - tt) * eps + tt * x
    v = model(xt, t)
    return ((v - (x - eps)) ** 2).mean()


def x1_from_v(xt: torch.Tensor, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return xt + (1 - t.view(-1, 1, 1, 1)) * v


@torch.no_grad()
def sample_uncond(model, shape, nfe: int = 50, device="cpu", solver: str = "euler", generator=None):
    """无条件 ODE 采样，用于检查先验质量。"""
    x = torch.randn(shape, device=device, generator=generator)
    ts = torch.linspace(0, 1, nfe + 1, device=device)
    for i in range(nfe):
        t0, t1 = ts[i], ts[i + 1]
        tb = t0.expand(shape[0])
        v = model(x, tb)
        if solver == "heun" and i < nfe - 1:
            x_pred = x + (t1 - t0) * v
            v2 = model(x_pred, t1.expand(shape[0]))
            x = x + (t1 - t0) * 0.5 * (v + v2)
        else:
            x = x + (t1 - t0) * v
    return x.clamp(-1, 1)
