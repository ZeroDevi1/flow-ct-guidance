"""Phase 1b 物理组件单元测试：python tests/test_dcfull.py"""
import math

import torch

from src.data.dataset import make_phantom
from src.physics.radon import CTOperator, interp_missing_rows, measured_mask, weighted_cg_dc


def _x(n=64, seed=3):
    return make_phantom(n, torch.Generator().manual_seed(seed))[None, None]


def test_measured_rows_match_sparse_operator():
    x = _x(); full = CTOperator(64, n_angles=180)
    for k, rng in ((60, math.pi), (30, math.pi), (120, 2 * math.pi / 3), (90, math.pi / 2)):
        m = measured_mask(180, k, rng)
        sparse = CTOperator(64, n_angles=k, angle_range=rng)(x)
        assert m.sum() == k
        assert torch.allclose(full(x)[..., m, :], sparse, atol=1e-4), (k, rng)


def test_misaligned_angles_rejected():
    try:
        measured_mask(180, 50)
    except ValueError:
        return
    raise AssertionError("50 视角不在 180 网格上，应报错")


def test_weighted_dc_reduces_to_m3_with_unit_weights():
    x = _x(); op = CTOperator(64, n_angles=180); y = op(x)
    x0 = torch.zeros_like(x)
    w = torch.ones(1, 180)
    a = weighted_cg_dc(op, x0, y, w, lam=0.3, n_iter=8, relative=True)
    b = op.cg_data_consistency(x0, y, lam=0.3, n_iter=8, relative=True)
    assert torch.allclose(a, b, atol=1e-5)


def test_zero_weight_rows_are_ignored():
    x = _x(); op = CTOperator(64, n_angles=180); y = op(x)
    m = measured_mask(180, 60)
    garbage = y.clone(); garbage[..., ~m, :] = 1e3  # 未测行是垃圾，但权重为 0
    w = m.float()[None]
    a = weighted_cg_dc(op, torch.zeros_like(x), garbage, w, lam=0.3, n_iter=8)
    b = weighted_cg_dc(op, torch.zeros_like(x), y, w, lam=0.3, n_iter=8)
    assert torch.allclose(a, b, atol=1e-4)


def test_interp_keeps_measured_rows():
    y = CTOperator(64, n_angles=180)(_x())
    for m in (measured_mask(180, 30), measured_mask(180, 120, 2 * math.pi / 3)):
        yf = y * m.view(1, 1, -1, 1)
        out = interp_missing_rows(yf, m)
        assert torch.equal(out[..., m, :], yf[..., m, :])
        assert (out[..., ~m, :] - y[..., ~m, :]).norm() < y[..., ~m, :].norm()  # 比全零好


if __name__ == "__main__":
    test_measured_rows_match_sparse_operator(); test_misaligned_angles_rejected()
    test_weighted_dc_reduces_to_m3_with_unit_weights(); test_zero_weight_rows_are_ignored()
    test_interp_keeps_measured_rows(); print("all dcfull tests passed")
