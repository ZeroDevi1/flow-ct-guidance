"""角度补全器单元测试：python tests/test_angle_ssm.py"""
import torch

from src.models.angle_ssm import AngleCompleter, count_params, selective_scan


def _scan_loop(u, delta, A, B, C):
    b, L, dm = u.shape
    h = u.new_zeros(b, dm, A.shape[1]); ys = []
    for k in range(L):
        a = (delta[:, k, :, None] * A).clamp(min=-5.0).exp()
        h = a * h + (delta[:, k] * u[:, k])[..., None] * B[:, k, None, :]
        ys.append((h * C[:, k, None, :]).sum(-1))
    return torch.stack(ys, 1)


def test_chunked_scan_matches_loop():
    g = torch.Generator().manual_seed(0)
    b, L, dm, n = 2, 37, 6, 4  # L 不是 chunk 的整数倍，覆盖尾块
    u = torch.randn(b, L, dm, generator=g); delta = torch.rand(b, L, dm, generator=g) * 2
    A = -torch.rand(dm, n, generator=g) * 3 - 0.1
    B = torch.randn(b, L, n, generator=g); C = torch.randn(b, L, n, generator=g)
    ref = _scan_loop(u, delta, A, B, C)
    out = selective_scan(u, delta, A, B, C, chunk=8)
    assert torch.allclose(out, ref, rtol=1e-4, atol=1e-4), (out - ref).abs().max()


def test_completer_shapes_mask_and_grad():
    a, d = 36, 64
    mask = torch.zeros(a, dtype=torch.bool); mask[::3] = True
    for mixer in ("ssm", "conv", "attn"):
        m = AngleCompleter(n_det=d, d=32, n_layers=2, mixer=mixer)
        s = torch.randn(2, 1, a, d); y = torch.randn(2, 1, a, d) * mask.view(1, 1, a, 1)
        yh, lv = m(s, y, mask, torch.rand(2))
        assert yh.shape == s.shape and lv.shape == (2, a)
        assert torch.equal(yh[..., mask, :], y[..., mask, :])  # 已测行强制等于测量
        # 零初始化解码头：初始输出在未测行等于 s_pred
        assert torch.allclose(yh[..., ~mask, :], s[..., ~mask, :])
        (yh.square().mean() + lv.mean()).backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())


def test_param_budget():
    m = AngleCompleter(n_det=256, d=256, n_layers=4, mixer="ssm")
    assert count_params(m) < 3.0, count_params(m)


if __name__ == "__main__":
    test_chunked_scan_matches_loop(); test_completer_shapes_mask_and_grad(); test_param_budget()
    for mx in ("ssm", "conv", "attn"):
        print(mx, f"{count_params(AngleCompleter(mixer=mx)):.2f}M")
    print("all angle_ssm tests passed")
