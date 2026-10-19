"""物理算子单元测试：python -m pytest tests/ 或直接 python tests/test_radon.py"""
import math
import torch
from src.physics.radon import CTOperator
from src.data.dataset import make_phantom


def _psnr(a, b, m):
    return 10 * torch.log10(4 / ((a - b)[m] ** 2).mean())


def test_adjoint():
    op = CTOperator(64, n_angles=45)
    x = torch.randn(2, 1, 64, 64); y = torch.randn(2, 1, 45, 64)
    lhs = (op.A_lin(x) * y).sum(); rhs = (x * op.A_lin_T(y)).sum()
    assert abs(lhs - rhs) / abs(lhs) < 1e-4


def test_fbp_quality():
    n = 128
    x = make_phantom(n, torch.Generator().manual_seed(1))[None, None]
    op = CTOperator(n, n_angles=180)
    m = op.mask.bool()
    assert _psnr(op.fbp(op(x)), x, m) > 30


def test_cg_improves_fbp():
    n = 128
    x = make_phantom(n, torch.Generator().manual_seed(2))[None, None]
    op = CTOperator(n, n_angles=45)
    m = op.mask.bool(); y = op(x)
    x0 = op.fbp(y)
    x1 = op.cg_data_consistency(x0, y, lam=0.01, n_iter=20).clamp(-1, 1)
    assert _psnr(x1, x, m) > _psnr(x0, x, m)


if __name__ == "__main__":
    test_adjoint(); test_fbp_quality(); test_cg_improves_fbp(); print("all radon tests passed")
