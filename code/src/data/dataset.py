"""切片数据集：读取 prepare_slices.py 产出的 npy。"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def data_root() -> Path:
    return Path(os.environ.get("MFP_DATA_ROOT", "/root/autodl-tmp/data"))


class SliceDataset(Dataset):
    """返回 [1, H, W] float32，取值 [-1, 1]。train 时做随机水平翻转。"""

    def __init__(self, processed_dir: str | Path, split: str = "train", augment: bool = True,
                 max_items: int | None = None, subsample: str = "head"):
        """subsample: max_items 生效时取前 N 张（"head"）或在整个划分上等间隔取 N 张（"even"，覆盖更多病人）。"""
        processed_dir = Path(processed_dir)
        self.x = np.load(processed_dir / f"slices_{split}.npy", mmap_mode="r")
        if max_items is not None and max_items < self.x.shape[0]:
            if subsample == "even":
                self.x = self.x[np.linspace(0, self.x.shape[0] - 1, max_items).round().astype(int)]
            else:
                self.x = self.x[:max_items]
        self.augment = augment and split == "train"

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        img = torch.from_numpy(np.asarray(self.x[i], dtype=np.float32))[None]
        if self.augment and torch.rand(()) < 0.5:
            img = img.flip(-1)
        return img


class SyntheticPhantomDataset(Dataset):
    """无数据时的冒烟测试集：随机椭圆相位体。只用于验证代码能跑通，不用于实验。"""

    def __init__(self, n: int = 256, size: int = 128, seed: int = 0):
        self.n, self.size = n, size
        self.g = torch.Generator().manual_seed(seed)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(i)
        return make_phantom(self.size, g)[None]


def make_phantom(size: int, g: torch.Generator | None = None, n_ellipses: int = 6) -> torch.Tensor:
    """随机椭圆叠加相位体，[-1, 1]，外圈为空气(-1)。"""
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing="ij")
    img = torch.full((size, size), -1.0)
    body = (xx ** 2 / 0.8 ** 2 + yy ** 2 / 0.6 ** 2) <= 1
    img[body] = -0.2
    for _ in range(n_ellipses):
        cx, cy = (torch.rand(2, generator=g) * 1.0 - 0.5).tolist()
        a, b = (torch.rand(2, generator=g) * 0.3 + 0.05).tolist()
        th = float(torch.rand((), generator=g)) * 3.14159
        val = float(torch.rand((), generator=g)) * 1.2 - 0.4
        xr = (xx - cx) * torch.cos(torch.tensor(th)) + (yy - cy) * torch.sin(torch.tensor(th))
        yr = -(xx - cx) * torch.sin(torch.tensor(th)) + (yy - cy) * torch.cos(torch.tensor(th))
        m = (xr ** 2 / a ** 2 + yr ** 2 / b ** 2) <= 1
        img[m & body] = val
    return img.clamp(-1, 1)
