"""HLCC 模块单元测试：python -m pytest tests/test_hlcc.py 或直接 python tests/test_hlcc.py"""
import torch

from src.data.dataset import make_phantom
from src.physics.hlcc import hlcc_project, hlcc_residual, legendre_basis
from src.physics.radon import CTOperator


def _sino(n=128, a=180, seed=1):
    x = make_phantom(n, torch.Generator().manual_seed(seed))[None, None]
    return CTOperator(n, n_angles=a)(x)


def test_legendre_orthogonal():
    P = legendre_basis(256, 6)
    G = (P @ P.T) / 256
    off = G - torch.diag(torch.diag(G))
    assert off.abs().max() < 1e-3  # 离散网格上近似正交


def test_true_sinogram_nearly_consistent():
    s = _sino()
    r_true = hlcc_residual(s)
    noisy = s + 0.05 * s.abs().mean() * torch.randn_like(s).cumsum(-2)  # 沿角度相关的扰动
    assert r_true < 0.2 * hlcc_residual(noisy), (r_true, hlcc_residual(noisy))


def test_projection_removes_low_order_bias_on_missing_rows():
    s = _sino()
    a, d = s.shape[-2:]
    measured = torch.zeros(a, dtype=torch.bool); measured[::3] = True  # 60/180 视角
    r = torch.linspace(-1, 1, d)
    bias = (0.3 + 0.2 * r) * s.abs().mean()  # 低阶偏差，只加在未测行
    corrupt = s.clone(); corrupt[..., ~measured, :] += bias
    out = hlcc_project(corrupt, measured)
    err_before = (corrupt - s)[..., ~measured, :].norm()
    err_after = (out - s)[..., ~measured, :].norm()
    assert torch.equal(out[..., measured, :], corrupt[..., measured, :])  # 已测行不变
    assert err_after < 0.3 * err_before, (err_before, err_after)
    assert hlcc_residual(out) < hlcc_residual(corrupt)


if __name__ == "__main__":
    test_legendre_orthogonal(); test_true_sinogram_nearly_consistent()
    test_projection_removes_low_order_bias_on_missing_rows(); print("all hlcc tests passed")
