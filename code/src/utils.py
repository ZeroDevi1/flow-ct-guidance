from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_uint8(x: torch.Tensor) -> np.ndarray:
    """[-1,1] -> uint8 灰度。"""
    return ((x.clamp(-1, 1) + 1) * 127.5).round().byte().cpu().numpy()


def save_grid(x: torch.Tensor, path: str | Path, nrow: int = 4, pad: int = 2):
    """x: [B,1,H,W] -> 拼成网格 PNG（不依赖 torchvision）。"""
    from PIL import Image
    x = x.detach().float()
    b, _, h, w = x.shape
    ncol = nrow; nr = (b + ncol - 1) // ncol
    canvas = np.full((nr * (h + pad) + pad, ncol * (w + pad) + pad), 255, np.uint8)
    arr = to_uint8(x)[:, 0]
    for i in range(b):
        r, c = divmod(i, ncol)
        y0, x0 = pad + r * (h + pad), pad + c * (w + pad)
        canvas[y0:y0 + h, x0:x0 + w] = arr[i]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(path)


def load_ema(path: str | Path, device="cpu"):
    from src.models.unet import UNet
    ck = torch.load(path, map_location=device)
    model = UNet(**ck["cfg"]["model"]).to(device)
    model.load_state_dict(ck["ema"]); model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ck["cfg"]


def save_per_image(out: str | Path, agg: dict[str, list[float]]):
    """逐图指标写 per_image.csv（列为 agg 中长度等于图像数的非空键），供按病人 bootstrap 与配对检验使用。"""
    n = len(agg["psnr"])
    keys = [k for k, v in agg.items() if len(v) == n]
    with open(Path(out) / "per_image.csv", "w") as f:
        f.write("idx," + ",".join(keys) + "\n")
        for i in range(n):
            f.write(f"{i}," + ",".join(f"{agg[k][i]:.6g}" for k in keys) + "\n")


def save_first(out: str | Path, arrs: list[torch.Tensor]):
    """保存前几张测试图的若干行图像（如 [GT, FBP, 重建, 标准差]，每行 [k,1,H,W]）为 first4.npy（float16）。"""
    np.save(Path(out) / "first4.npy", torch.stack(arrs).squeeze(2).detach().float().cpu().numpy().astype(np.float16))
