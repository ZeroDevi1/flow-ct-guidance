"""PSNR / SSIM / 数据一致性。所有指标在 [-1,1] 归一化域计算（等价于 HU 窗 [-1000,1000]）。"""
from __future__ import annotations

import numpy as np
import torch


def psnr(x: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """逐样本 PSNR，动态范围 2.0。返回 [B]。"""
    d = (x - ref) ** 2
    if mask is not None:
        d = d * mask
        mse = d.flatten(1).sum(1) / mask.flatten(1).sum(1).clamp(min=1)
    else:
        mse = d.flatten(1).mean(1)
    return 10 * torch.log10(4.0 / mse.clamp(min=1e-12))


def ssim(x: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """逐样本 SSIM（skimage 实现，数据范围 2.0）。返回 [B]。

    mask 给定时只在掩膜内平均 SSIM 图：重建域外恒为 -1（空气），不掩膜会把所有方法的 SSIM 一起抬高。
    """
    from skimage.metrics import structural_similarity
    out = []
    xn, rn = x.detach().float().cpu().numpy(), ref.detach().float().cpu().numpy()
    m = None if mask is None else mask.detach().cpu().numpy().reshape(xn.shape[-2:]) > 0.5
    for i in range(x.shape[0]):
        if m is None:
            out.append(structural_similarity(xn[i, 0], rn[i, 0], data_range=2.0))
        else:
            _, smap = structural_similarity(xn[i, 0], rn[i, 0], data_range=2.0, full=True)
            out.append(float(smap[m].mean()))
    return torch.tensor(out)


_LPIPS = {}


def lpips_dist(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """逐样本 LPIPS(AlexNet)，输入 [-1,1] 单通道，复制为三通道。首次调用加载网络并缓存。返回 [B]。"""
    import lpips
    dev = x.device
    if dev not in _LPIPS:
        _LPIPS[dev] = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
    with torch.no_grad():
        d = _LPIPS[dev](x.float().clamp(-1, 1).repeat(1, 3, 1, 1), ref.float().repeat(1, 3, 1, 1))
    return d.flatten().cpu()


def data_consistency(op, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """||A x - y|| / ||y||，返回 [B]。"""
    r = op(x) - y
    return r.flatten(1).norm(dim=1) / y.flatten(1).norm(dim=1).clamp(min=1e-12)


def summarize(d: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    return {k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)} for k, v in d.items() if v}
