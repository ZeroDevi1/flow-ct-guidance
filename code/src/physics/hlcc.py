"""平行束正弦图的 Helgason–Ludwig 一致性条件（HLCC）：残差度量与加权投影。

背景
----
对平行束 Radon 变换 s(θ, r) = ∫ f(r·e_θ + τ·e_θ⊥) dτ，n 阶矩
    M_n(θ) = ∫ s(θ, r) rⁿ dr
必须是 (cos θ, sin θ) 的 n 次齐次多项式，等价地：其 Fourier 级数只含谐波 |m| ≤ n 且 m ≡ n (mod 2)。
任何真实扫描的正弦图都满足这组条件；学习式补全得到的正弦图不一定满足。

离散约定（与 radon.py 一致）
----------------------------
* 正弦图 [B, 1, A, D]：A 个角度均匀分布在 [0, π)，D 个探测器单元覆盖 r ∈ [−1, 1]。
* 用周期延拓 s(θ+π, r) = s(θ, −r) 得到 [0, 2π) 上的 2A 行，矩在延拓序列上满足 M_n(θ+π) = (−1)ⁿ M_n(θ)。
* 为了数值稳定，用 Legendre 多项式 P_n(r) 代替 rⁿ：P_n 的矩也满足同样的谐波约束（它是 r⁰..rⁿ 的线性组合，
  而 rᵏ 的矩只含 |m| ≤ k、m ≡ k 的谐波，与 n 同奇偶的项保留，奇偶相反的项系数为 0）。
* 离散化（双线性旋转 + 求和）使真实正弦图的残差不为 0，而是一个小的底噪；单元测试检查的是
  "真实正弦图残差 ≪ 扰动后残差" 以及 "投影使残差下降"。
"""
from __future__ import annotations

import math

import torch


_BASIS_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
_HARM_CACHE: dict[tuple, tuple[list[torch.Tensor], torch.Tensor]] = {}


def legendre_basis(d: int, n_max: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """返回 [n_max+1, d] 的 Legendre 多项式在探测器网格 r_j = −1 + (2j+1)/d 上的取值（Bonnet 递推）。"""
    r = (torch.arange(d, device=device, dtype=dtype) * 2 + 1) / d - 1
    P = [torch.ones_like(r), r]
    for n in range(1, n_max):
        P.append(((2 * n + 1) * r * P[n] - n * P[n - 1]) / (n + 1))
    return torch.stack(P[: n_max + 1])


def _get_basis(d: int, n_max: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """缓存 Legendre 基 P [N+1, D] 与最小范数反投影矩阵 Pᵀ(PPᵀ)⁻¹ [D, N+1]。"""
    key = (d, n_max, str(device), dtype)
    if key not in _BASIS_CACHE:
        P = legendre_basis(d, n_max, device, dtype)
        # P_pinv = Pᵀ (P Pᵀ)⁻¹，满足 P @ P_pinv = I
        P_pinv = torch.linalg.solve(P @ P.T, P).T
        _BASIS_CACHE[key] = (P, P_pinv)
    return _BASIS_CACHE[key]


def _harmonic_design(a: int, n: int, device, dtype) -> torch.Tensor:
    """[2A, K] 设计矩阵：在 2A 个均匀角度 θ∈[0,2π) 上，阶 n 允许的 cos/sin 谐波（|m|≤n, m≡n mod 2）。"""
    theta = torch.arange(2 * a, device=device, dtype=dtype) * (math.pi / a)
    cols = []
    for m in range(n % 2, n + 1, 2):
        if m == 0:
            cols.append(torch.ones_like(theta))
        else:
            cols += [torch.cos(m * theta), torch.sin(m * theta)]
    return torch.stack(cols, dim=1)


def _get_harmonics(a: int, n_max: int, device, dtype) -> tuple[list[torch.Tensor], torch.Tensor]:
    """缓存各阶谐波设计矩阵 X_n [2A, K_n] 与无权重正交投影矩阵 proj [N+1, 2A, 2A]。"""
    key = (a, n_max, str(device), dtype)
    if key not in _HARM_CACHE:
        Xs = [_harmonic_design(a, n, device, dtype) for n in range(n_max + 1)]
        projs = []
        for X in Xs:
            Q, _ = torch.linalg.qr(X)
            projs.append(Q @ Q.T)
        _HARM_CACHE[key] = (Xs, torch.stack(projs))
    return _HARM_CACHE[key]


def extend_periodic(s: torch.Tensor) -> torch.Tensor:
    """[B,1,A,D] → [B,1,2A,D]：后半段为 s(θ+π, r) = s(θ, −r)。"""
    return torch.cat([s, s.flip(-1)], dim=-2)


def moments(s: torch.Tensor, n_max: int) -> torch.Tensor:
    """延拓后正弦图的 Legendre 矩：[B, n_max+1, 2A]。"""
    P, _ = _get_basis(s.shape[-1], n_max, s.device, s.dtype)  # [N+1, D]
    return torch.einsum("bad,nd->bna", extend_periodic(s)[:, 0], P)


def _fit(M: torch.Tensor, w: torch.Tensor | None, n_max: int, ridge: float = 1e-4) -> torch.Tensor:
    """对每阶矩做（加权）最小二乘谐波拟合，返回拟合值 [B, N+1, 2A]。w: [B, 2A] 或 None。"""
    b, _, a2 = M.shape
    Xs, proj = _get_harmonics(a2 // 2, n_max, M.device, M.dtype)
    if w is None:
        # 固定正交投影：零 lstsq 开销，训练损失完全可微
        return torch.einsum("naq,bnq->bna", proj, M)
    out = torch.empty_like(M)
    w_clamped = w.clamp(min=1e-6)  # [B, 2A]
    # 岭正则按权重量级归一化，防止限角大缺口下高阶谐波（如 n=6）病态外推振荡
    reg_scale = ridge * w_clamped.sum(dim=1, keepdim=True)  # [B, 1]
    for n, X in enumerate(Xs):
        k = X.shape[1]
        wx = w_clamped[:, :, None] * X[None]  # [B, 2A, K]
        xtwx = torch.einsum("ai,baj->bij", X, wx)  # [B, K, K]
        eye = torch.eye(k, device=M.device, dtype=M.dtype)[None]
        rhs = torch.einsum("bai,ba->bi", wx, M[:, n])[..., None]  # [B, K, 1]
        coef = torch.linalg.solve(xtwx + reg_scale[:, :, None] * eye, rhs)  # [B, K, 1]
        out[:, n] = torch.einsum("ak,bko->ba", X, coef)
    return out


def hlcc_residual(s: torch.Tensor, n_max: int = 6) -> torch.Tensor:
    """相对 HLCC 残差：‖M − 拟合(M)‖ / ‖M‖，逐样本 [B]。用于单元测试、训练损失与诊断。"""
    M = moments(s, n_max)
    R = M - _fit(M, None, n_max)
    return R.flatten(1).norm(dim=1) / M.flatten(1).norm(dim=1).clamp(min=1e-12)


def hlcc_project(s: torch.Tensor, measured: torch.Tensor, weight: torch.Tensor | None = None,
                 n_max: int = 6, ridge: float = 1e-4) -> torch.Tensor:
    """把正弦图的未测行修正到满足 HLCC，已测行保持不变。

    参数
        s:        [B,1,A,D] 补全后的正弦图（已测行应等于测量值）。
        measured: [A] bool，哪些角度是实测的（所有样本共用一种采样模式）。
        weight:   [B,A] 拟合权重（已测行通常为 1，未测行为补全器置信度）；None 时已测行 1、未测行 0.1。
        ridge:    加权谐波拟合的相对岭正则系数（限角大缺口时抑制高阶振荡）。
    返回
        修正后的正弦图 [B,1,A,D]。对每个未测角度 θ，加上最小范数修正 δ(θ,·)，使其 0..n_max 阶矩等于加权
        谐波拟合值；已测行逐元素不变。2A 延拓的后半段与前半段对应同一组物理测量，修正只取前半段。
    """
    b, _, a, d = s.shape
    _, P_pinv = _get_basis(d, n_max, s.device, s.dtype)  # [D, N+1]
    M = moments(s, n_max)  # [B, N+1, 2A]
    if weight is None:
        weight = torch.where(measured, 1.0, 0.1).to(s.dtype).expand(b, a)
    fit = _fit(M, torch.cat([weight, weight], dim=1), n_max, ridge=ridge)
    dM = (fit - M)[..., :a]  # 只修正前半段对应的物理行 [B, N+1, A]
    delta = torch.einsum("dn,bna->bad", P_pinv, dM)
    delta = delta * (~measured).to(s.dtype)[None, :, None]
    return s + delta[:, None]
