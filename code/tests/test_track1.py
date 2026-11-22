"""Track 1 物理与采样组件单元测试：python tests/test_track1.py"""
import torch

from src.data.dataset import make_phantom
from src.physics.radon import (CTOperator, diag_aat_approx, diag_atwa_approx,
                               poisson_ray_variance, weighted_cg_dc)
from src.sample_posterior import posterior_sample


class DummyFlow(torch.nn.Module):
    """恒等速度场测试桩：v(x, t) = -x（将 x_t 拉向 0）。"""

    def forward(self, x, t):
        return -0.5 * x


def _x(n=64, seed=7):
    return make_phantom(n, torch.Generator().manual_seed(seed))[None, None]


def test_poisson_ray_variance_monotonic():
    y = torch.tensor([[[[0.0, 50.0, 100.0]]]])
    var_1e5 = poisson_ray_variance(y, i0=1e5, sigma=0.0, mu_scale=0.06)
    var_1e4 = poisson_ray_variance(y, i0=1e4, sigma=0.0, mu_scale=0.06)
    assert var_1e5[0, 0, 0, 0] < var_1e5[0, 0, 0, 1] < var_1e5[0, 0, 0, 2]
    assert torch.allclose(var_1e4, var_1e5 * 10.0, rtol=1e-4)


def test_weighted_cg_dc_4d_and_precond():
    x = _x()
    op = CTOperator(64, n_angles=60)
    y = op(x)
    y_noisy = op.add_noise(y, i0=1e4, mu_scale=0.06, generator=torch.Generator().manual_seed(0))
    var = poisson_ray_variance(y_noisy, i0=1e4, mu_scale=0.06)
    w = 1.0 / var
    diag = diag_atwa_approx(op, w)
    assert diag.shape == x.shape and (diag >= 0).all()
    x0 = torch.zeros_like(x)
    # 检验预条件 CG 在相同迭代步数下残差显著低于无预条件 CG，且多步下收敛到极低残差
    rhs = op.A_lin_T(w * (y_noisy - op.offset)) + 1.0 * x0
    def res_norm(sol):
        return (op.A_lin_T(w * op.A_lin(sol)) + 1.0 * sol - rhs).norm()
    a10 = weighted_cg_dc(op, x0, y_noisy, w, lam=1.0, n_iter=10, relative=False, precond_diag=diag)
    b10 = weighted_cg_dc(op, x0, y_noisy, w, lam=1.0, n_iter=10, relative=False, precond_diag=None)
    assert res_norm(a10) < res_norm(b10), f"PCG 10-iter residual {res_norm(a10):.3e} >= CG {res_norm(b10):.3e}"
    a60 = weighted_cg_dc(op, x0, y_noisy, w, lam=1.0, n_iter=60, relative=False, precond_diag=diag)
    assert res_norm(a60) / rhs.norm() < 5e-4, f"PCG 60-iter rel residual {res_norm(a60)/rhs.norm():.3e}"
    # 4D 全 1 权重应退化为标准 cg_data_consistency
    c = weighted_cg_dc(op, x0, y, torch.ones_like(y), lam=0.3, n_iter=10, relative=True)
    d = op.cg_data_consistency(x0, y, lam=0.3, n_iter=10, relative=True)
    assert torch.allclose(c, d, atol=1e-4)


def test_diag_aat_approx_scale():
    op = CTOperator(64, n_angles=60)
    d = diag_aat_approx(op)
    assert d.shape == op.offset.shape and (d >= 0).all()
    assert abs(float(d.mean()) - op.aat_scale) / op.aat_scale < 1e-4


def test_posterior_sample_track1_modes():
    x = _x()
    op = CTOperator(64, n_angles=60)
    y = op.add_noise(op(x), i0=1e5, mu_scale=0.06, generator=torch.Generator().manual_seed(1))
    model = DummyFlow()
    phys = {"i0": 1e5, "sigma": 0.0, "mu_scale": 0.06}

    for kw in (
        dict(guidance="pigdm", noise_model="scalar", physics_cfg=phys),
        dict(guidance="pigdm", noise_model="poisson", pigdm_cg_iters=5, physics_cfg=phys),
        dict(guidance="dc", noise_model="poisson", dc_schedule="const", dc_iters=5, physics_cfg=phys),
        dict(guidance="dc", noise_model="poisson", dc_schedule="rt2", dc_iters=5, physics_cfg=phys),
        dict(guidance="hybrid", t_switch=0.4, noise_model="none", physics_cfg=phys),
    ):
        out = posterior_sample(model, op, y, x.shape, nfe=5, **kw)
        assert out.shape == x.shape and torch.isfinite(out).all(), kw


if __name__ == "__main__":
    test_poisson_ray_variance_monotonic()
    test_weighted_cg_dc_4d_and_precond()
    test_diag_aat_approx_scale()
    test_posterior_sample_track1_modes()
    print("all track1 tests passed")
