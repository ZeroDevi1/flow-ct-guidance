"""用训练好的流匹配先验做 CT 反问题的后验采样（M1/M2/M3）。

三种把测量 y 引入 ODE 采样的方式（同一先验、同一 NFE，便于公平比较）：
    dps   : Euler 步后减去 Δt·ζ·∇_x ||A(x̂₁) - y|| / √s（DPS 风格，梯度穿过网络；s = tr(AᵀA)/N²）
    pigdm : ΠGDM 风格伪逆引导：vjp(x̂₁, Aᵀ(AAᵀ + r_t² I)⁻¹(y - A x̂₁))，CG 近似求逆
    dc    : 每步对 x̂₁ 做共轭梯度数据一致性，再按 (ε̂, x̂₁) 重组 x_{t+Δt}（投影式）
    dcfull: Phase 1b。在 180 角全网格上做逐角度加权 DC：未测行由补全器（角度 SSM / 环形插值 / 先验预测）
            给出，可选 HLCC 投影；设计见 docs/02_phase1b_angle_ssm_design.md

用法：
    python src/sample_posterior.py --ckpt checkpoints/prior_256/ema_final.pt \
        --config configs/eval_sparse60.yaml --guidance dps --out results/w2_dps_sv60
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml

from src.data.dataset import SliceDataset, SyntheticPhantomDataset
from src.eval.metrics import data_consistency, lpips_dist, psnr, ssim, summarize
from src.physics.hlcc import hlcc_project
from src.physics.radon import (CTOperator, diag_aat_approx, diag_atwa_approx, interp_missing_rows,
                               measured_mask, poisson_ray_variance, weighted_cg_dc)
from src.utils import load_ema, save_first, save_grid, save_per_image, set_seed


def build_operator(cfg, n, device):
    p = cfg["physics"]
    return CTOperator(n, n_angles=p.get("n_angles", 180), angle_range=p.get("angle_range", 3.141592653589793),
                      device=device, angle_offset=p.get("angle_offset", 0.0))


def make_measurement(op, x, cfg, generator=None, indices=None):
    """生成测量 y = A x + n。

    indices: 测试集图像下标。给定时每张图的噪声用独立种子 (seed, 下标) 生成，
    使同一张测试图在不同方法 / batch 大小 / NFE / 采样次数下拿到完全相同的 y；
    训练（FBPConvNet）时不给 indices，用传入的 generator 产生随机噪声。
    """
    y = op(x)
    p = cfg["physics"]
    kw = dict(i0=p.get("i0"), sigma=p.get("sigma", 0.0), mu_scale=p.get("mu_scale", 0.06))
    if indices is None:
        return op.add_noise(y, generator=generator, **kw)
    base = 1_000_003 * (cfg.get("seed", 0) + 1)
    out = []
    for k, idx in enumerate(indices):
        g = torch.Generator(device=y.device).manual_seed(base + int(idx))
        out.append(op.add_noise(y[k:k + 1], generator=g, **kw))
    return torch.cat(out)


def posterior_sample(model, op, y, shape, guidance: str, nfe: int, zeta: float = 1.0, dc_lam: float = 1.0,
                     dc_iters: int = 5, pigdm_cg_iters: int = 5, generator=None, t_start_guidance: float = 0.0,
                     pigdm_sigma_y: float = 0.0, pigdm_eta: float = 1e-3, full: dict | None = None,
                     t_switch: float = 0.4, noise_model: str = "none", noise_scale: float = 1.0,
                     dc_schedule: str = "const", physics_cfg: dict | None = None):
    """从 t=0（噪声）积分到 t=1（数据），每步按 guidance 引入测量 y。

    dps   : x ← x + Δt·(v − ζ·∇ₓ‖A x̂₁ − y‖/√s)，s = tr(AᵀA)/N²；ζ 乘 Δt 使总引导量与 NFE 无关，
            除以 √s 使 ζ 与角度数无关。
    pigdm : 解 (AAᵀ + (Σ_y/r_t² + η·s I)) z = y − A x̂₁（s = tr(AAᵀ)/m），
            x ← x + Δt·v + ζ·t₁·∇ₓ⟨x̂₁, Aᵀz⟩；t₁ 对应 ΠGDM 中的信号系数 √ᾱ_t。
            noise_model="scalar" 用解析均值方差 σ̄_y²，"poisson" 用逐射线方差 Σ_y + Jacobi 预条件 CG。
    dc    : x̂₁ 做 CG 近端投影 argmin ‖W^{1/2}(A x − y)‖² + λ(t)‖x − x̂₁‖²，再按 (ε̂, x̂₁_dc) 重组 x_{t+Δt}。
            noise_model="poisson" 时 W = Σ_y⁻¹ ∝ I₀ e^{-μy}；dc_schedule="rt2" 时 λ(t) = λ₀/r_t²（精确 MAP）。
    hybrid: 混合调度，t < t_switch 用 M3(dc) 稳定全局结构，t ≥ t_switch 切 M2(pigdm) 锐化边缘。
    dcfull: Phase 1b。full 为上下文字典 {op, mask, y_full, mode, completer, y_interp, kappa, use_conf, hlcc}。
    """
    device = y.device
    x = torch.randn(shape, device=device, generator=generator)
    ts = torch.linspace(0, 1, nfe + 1, device=device)
    b = shape[0]

    # 预先按物理噪声模型构造逐射线方差与对角预条件器（仅算一次，避免在 ODE 循环内重复求伴随）
    sigma_y2_ray = None
    sigma_y2_scalar = None
    w_ray = None
    diag_atwa = None
    diag_aat = None
    if noise_model != "none" and physics_cfg is not None:
        i0 = physics_cfg.get("i0")
        sigma = float(physics_cfg.get("sigma", 0.0) or 0.0)
        mu_scale = float(physics_cfg.get("mu_scale", 0.06))
        if (i0 is not None and i0 > 0) or sigma > 0:
            sigma_y2_ray = poisson_ray_variance(y, i0=i0, sigma=sigma, mu_scale=mu_scale)
            sigma_y2_scalar = sigma_y2_ray.flatten(1).mean(dim=1).view(-1, 1, 1, 1)
            if noise_model == "poisson":
                w_ray = 1.0 / sigma_y2_ray.clamp(min=1e-12)
                if guidance in ("dc", "hybrid"):
                    diag_atwa = diag_atwa_approx(op, w_ray)
                if guidance in ("pigdm", "hybrid"):
                    diag_aat = diag_aat_approx(op)

    for i in range(nfe):
        t0, t1 = ts[i], ts[i + 1]
        tb = t0.expand(b)
        dt = t1 - t0
        step_guid = ("dc" if float(t0) < t_switch else "pigdm") if guidance == "hybrid" else guidance
        need_grad = step_guid in ("dps", "pigdm") and float(t0) >= t_start_guidance
        with torch.set_grad_enabled(need_grad):
            if need_grad:
                x = x.detach().requires_grad_(True)
            v = model(x, tb)
            x1 = x + (1 - t0) * v
            if step_guid == "dps" and need_grad:
                r = op(x1) - y
                norm = r.flatten(1).norm(dim=1)
                g, = torch.autograd.grad(norm.sum(), x)
                # 梯度量级随算子范数 √tr(AᵀA)/N 增长（60 视角约 3.4，180 视角约 5.9），除以它使 ζ 跨视角数可比
                x_next = (x + dt * (v - zeta * g / op.ata_scale ** 0.5)).detach()
            elif step_guid == "pigdm" and need_grad:
                r = (y - op(x1)).detach()
                # x_t=(1-t)ε+t x 下 x̂₁ 的后验方差系数 r_t² = (1-t)²/(t²+(1-t)²)（等效 VE σ̃=(1-t)/t）
                r_t2 = float(((1 - t0) ** 2) / (t0 ** 2 + (1 - t0) ** 2 + 1e-8))
                if noise_model == "poisson" and sigma_y2_ray is not None:
                    damp = (noise_scale * sigma_y2_ray) / max(r_t2, 1e-8) + pigdm_eta * op.aat_scale
                elif noise_model == "scalar" and sigma_y2_scalar is not None:
                    damp = (noise_scale * sigma_y2_scalar) / max(r_t2, 1e-8) + pigdm_eta * op.aat_scale
                else:
                    damp = pigdm_sigma_y ** 2 / max(r_t2, 1e-8) + pigdm_eta * op.aat_scale
                m_inv = None if (diag_aat is None or noise_model != "poisson") else 1.0 / (diag_aat + damp).clamp(min=1e-8)
                # 测量域（预条件）CG 解 (A Aᵀ + damp) z = r
                z = torch.zeros_like(r); res = r.clone()
                u = res if m_inv is None else m_inv * res
                p = u.clone()
                rz = (res * u).flatten(1).sum(1)
                for _ in range(pigdm_cg_iters):
                    Ap = op.A_lin(op.A_lin_T(p)) + damp * p
                    alpha = rz / ((p * Ap).flatten(1).sum(1) + 1e-12)
                    z = z + alpha.view(-1, 1, 1, 1) * p
                    res = res - alpha.view(-1, 1, 1, 1) * Ap
                    u = res if m_inv is None else m_inv * res
                    rz_new = (res * u).flatten(1).sum(1)
                    p = u + (rz_new / (rz + 1e-12)).view(-1, 1, 1, 1) * p
                    rz = rz_new
                w = op.A_lin_T(z).detach()
                g, = torch.autograd.grad((x1 * w).sum(), x)
                x_next = (x + dt * v + zeta * t1 * g).detach()
            else:
                x_next = (x + dt * v).detach()
        if step_guid == "dcfull" and float(t0) >= t_start_guidance:
            with torch.no_grad():
                x1 = x1.detach()
                eps_hat = x.detach() - t0 * v.detach()
                x1_dc = _dcfull_step(full, x1, t0, b, dc_lam, dc_iters)
                x_next = (1 - t1) * eps_hat + t1 * x1_dc
        if step_guid == "dc" and float(t0) >= t_start_guidance:
            with torch.no_grad():
                x1 = x1.detach()
                eps_hat = x.detach() - t0 * v.detach()
                r_t2 = float(((1 - t0) ** 2) / (t0 ** 2 + (1 - t0) ** 2 + 1e-8))
                if noise_model == "poisson" and w_ray is not None:
                    if dc_schedule == "rt2":
                        lam_step = (dc_lam * noise_scale) / max(r_t2, 1e-8)
                        x1_dc = weighted_cg_dc(op, x1, y, w_ray, lam=lam_step, n_iter=dc_iters,
                                               relative=False, precond_diag=diag_atwa)
                    else:
                        x1_dc = weighted_cg_dc(op, x1, y, w_ray, lam=dc_lam, n_iter=dc_iters,
                                               relative=True, precond_diag=diag_atwa)
                elif dc_schedule == "rt2":
                    lam_step = dc_lam / max(r_t2, 1e-8)
                    x1_dc = op.cg_data_consistency(x1, y, lam=lam_step, n_iter=dc_iters, relative=True)
                else:
                    x1_dc = op.cg_data_consistency(x1, y, lam=dc_lam, n_iter=dc_iters, relative=True)
                x_next = (1 - t1) * eps_hat + t1 * x1_dc
        x = x_next
    return x.detach().clamp(-1, 1)


def _dcfull_step(full: dict, x1: torch.Tensor, t0: torch.Tensor, b: int, dc_lam: float, dc_iters: int):
    """dcfull 的一步：构造全角度伪测量 ŷ 与逐行权重 w，可选 HLCC 投影，再做加权近端 DC。"""
    op, mask, y_full = full["op"], full["mask"], full["y_full"]
    s_pred = op(x1)
    a = mask.numel()
    w_un = torch.full((b, a), float(full["kappa"]), device=x1.device)
    if full["mode"] == "ssm":
        y_hat, logvar = full["completer"](s_pred, y_full, mask, t0.expand(b))
        if full["use_conf"]:
            # 按设计公式 w_θ = κ · σ̄² / σ²_θ（相对未测行均值方差归一化），避免 logvar≪0 时权重全部饱和到 1.0
            sigma2 = logvar.clamp(-10.0, 4.0).exp()
            un = ~mask
            if un.any():
                sigma2_bar = sigma2[:, un].mean(dim=1, keepdim=True).clamp(min=1e-8)
                w_un = (full["kappa"] * (sigma2_bar / sigma2.clamp(min=1e-8))).clamp(min=1e-4, max=1.0)
    elif full["mode"] == "interp":
        y_hat = full["y_interp"]
    elif full["mode"] == "pred":
        y_hat = torch.where(mask.view(1, 1, a, 1), y_full, s_pred)
    else:
        raise ValueError(full["mode"])
    w = torch.where(mask.view(1, a), torch.ones_like(w_un), w_un)
    if full["hlcc"]:
        y_hat = hlcc_project(y_hat, mask, weight=w, n_max=full.get("hlcc_nmax", 6))
    return weighted_cg_dc(op, x1, y_hat, w, lam=dc_lam, n_iter=dc_iters, relative=True)


def build_full_context(cfg, n, device, y, mode, completer_ckpt=None, kappa=0.3, use_conf=True, use_hlcc=False,
                       hlcc_nmax: int = 6, op_full: CTOperator | None = None, mask: torch.Tensor | None = None):
    """为 dcfull 构造上下文：180 角全算子、已测掩码、全网格上的 y、可选补全器 / 插值结果。"""
    p = cfg["physics"]
    if mask is None:
        mask = measured_mask(180, p.get("n_angles", 180), p.get("angle_range", 3.141592653589793)).to(device)
    op = op_full if op_full is not None else CTOperator(n, n_angles=180, device=device)
    y_full = y.new_zeros(y.shape[0], 1, 180, y.shape[-1])
    y_full[:, :, mask] = y
    ctx = {"op": op, "mask": mask, "y_full": y_full, "mode": mode, "kappa": kappa, "use_conf": use_conf,
           "hlcc": use_hlcc, "hlcc_nmax": hlcc_nmax, "completer": None, "y_interp": None}
    if mode == "interp":
        ctx["y_interp"] = interp_missing_rows(y_full, mask)
    if mode == "ssm":
        ctx["completer"] = completer_ckpt
    return ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--guidance", choices=["dps", "pigdm", "dc", "hybrid", "dcfull", "none"], default="dps")
    ap.add_argument("--t-switch", type=float, default=0.4, help="hybrid 调度从 M3(dc) 切换到 M2(pigdm) 的时刻")
    ap.add_argument("--noise-model", choices=["none", "scalar", "poisson"], default="none",
                    help="噪声感知模式：none（原无噪假设）、scalar（解析均值方差）、poisson（逐射线逆方差 + PCG）")
    ap.add_argument("--noise-scale", type=float, default=1.0, help="解析噪声方差标定倍率（默认 1.0）")
    ap.add_argument("--dc-schedule", choices=["const", "rt2"], default="const",
                    help="dc 的 λ 时间调度：const（常数）或 rt2（精确 MAP 的 λ₀/r_t²）")
    ap.add_argument("--dc-iters", type=int, default=None, help="覆盖 sampling.dc_iters")
    ap.add_argument("--pigdm-cg-iters", type=int, default=None, help="覆盖 sampling.pigdm_cg_iters")
    ap.add_argument("--pigdm-sigma-y", type=float, default=None, help="覆盖 sampling.pigdm_sigma_y")
    ap.add_argument("--full-mode", choices=["ssm", "interp", "pred"], default="ssm", help="dcfull 的未测行来源")
    ap.add_argument("--completer", default=None, help="dcfull/ssm 用的补全器权重（train_completer.py 输出）")
    ap.add_argument("--kappa", type=float, default=0.3, help="dcfull 未测行的基准权重")
    ap.add_argument("--no-conf", action="store_true", help="dcfull/ssm 不用预测置信度，未测行权重固定为 κ")
    ap.add_argument("--hlcc", action="store_true", help="dcfull 中对 ŷ 做 HLCC 投影")
    ap.add_argument("--hlcc-nmax", type=int, default=6, help="HLCC 投影最大矩阶数")
    ap.add_argument("--out", required=True)
    ap.add_argument("--nfe", type=int, default=None)
    ap.add_argument("--zeta", type=float, default=None)
    ap.add_argument("--n-samples", type=int, default=None, help="每张测试图采样次数（不确定性）")
    ap.add_argument("--n-test", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None, help="覆盖 sampling.batch_size")
    ap.add_argument("--data-dir", default=None, help="覆盖 data.processed_dir")
    ap.add_argument("--dc-lam", type=float, default=None, help="覆盖 sampling.dc_lam（相对量纲）")
    ap.add_argument("--split", default=None, help="覆盖 data.split；调参用 val，主表用 test200")
    ap.add_argument("--t-start", type=float, default=None, help="覆盖 sampling.t_start_guidance：t 小于该值的步不加引导")
    ap.add_argument("--no-lpips", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.data_dir:
        cfg["data"]["processed_dir"] = args.data_dir
    s = cfg["sampling"]
    nfe = args.nfe or s.get("nfe", 50)
    zeta = args.zeta if args.zeta is not None else s.get("zeta", 1.0)
    dc_lam = args.dc_lam if args.dc_lam is not None else s.get("dc_lam", 1.0)
    dc_iters = args.dc_iters if args.dc_iters is not None else s.get("dc_iters", 5)
    pigdm_cg_iters = args.pigdm_cg_iters if args.pigdm_cg_iters is not None else s.get("pigdm_cg_iters", 5)
    pigdm_sigma_y = args.pigdm_sigma_y if args.pigdm_sigma_y is not None else s.get("pigdm_sigma_y", 0.0)
    split = args.split or cfg["data"].get("split", "test")
    t_guid = args.t_start if args.t_start is not None else s.get("t_start_guidance", 0.0)
    n_samples = args.n_samples or s.get("n_samples", 1)
    n_test = args.n_test or cfg["data"].get("n_test", 200)
    bs = args.batch_size or s.get("batch_size", 8)
    set_seed(cfg.get("seed", 0))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    model, mcfg = load_ema(args.ckpt, device)
    completer = None
    if args.guidance == "dcfull" and args.full_mode == "ssm":
        from src.models.angle_ssm import AngleCompleter
        ck = torch.load(args.completer, map_location=device)
        # 探测器单元数与先验分辨率一致（训练时同样以 n_det=image_size 构造）
        completer = AngleCompleter(n_det=mcfg["model"]["image_size"], **ck["cfg"]["model"]).to(device).eval()
        completer.load_state_dict(ck["model"])
        for p_ in completer.parameters():
            p_.requires_grad_(False)
    n = mcfg["model"]["image_size"]
    op = build_operator(cfg, n, device)
    if cfg["data"]["type"] == "phantom":
        ds = SyntheticPhantomDataset(n=n_test, size=n, seed=123)
    else:
        # val 前 N 张只来自一两个病人，调参时改为在整个划分上等间隔取样
        ds = SliceDataset(cfg["data"]["processed_dir"], split=split, augment=False, max_items=n_test,
                          subsample="even" if split == "val" else "head")
    # 采样初始噪声的随机流；测量噪声按图像下标独立播种（见 make_measurement），两者互不影响
    gen = torch.Generator(device=device).manual_seed(cfg.get("seed", 0))
    use_lpips = not args.no_lpips

    agg = {"psnr": [], "ssim": [], "lpips": [], "dc": [], "psnr_fbp": [], "ssim_fbp": [], "lpips_fbp": [],
           "psnr_mean_of_samples": [], "pixel_std": []}
    # 复用 180 角全算子与掩码，避免每个 batch 重复构造旋转网格与标定
    op_full, full_mask = None, None
    if args.guidance == "dcfull":
        p = cfg["physics"]
        full_mask = measured_mask(180, p.get("n_angles", 180), p.get("angle_range", 3.141592653589793)).to(device)
        op_full = CTOperator(n, n_angles=180, device=device)
    first_batch, worst = None, []
    t_start = time.time()
    for i0 in range(0, len(ds), bs):
        idx = list(range(i0, min(i0 + bs, len(ds))))
        x = torch.stack([ds[j] for j in idx]).to(device)
        y = make_measurement(op, x, cfg, indices=idx)
        x_fbp = op.fbp(y)
        full = None
        if args.guidance == "dcfull":
            full = build_full_context(cfg, n, device, y, args.full_mode, completer, kappa=args.kappa,
                                      use_conf=not args.no_conf, use_hlcc=args.hlcc, hlcc_nmax=args.hlcc_nmax,
                                      op_full=op_full, mask=full_mask)
        samples = []
        for _ in range(n_samples):
            xs = posterior_sample(model, op, y, x.shape, args.guidance, nfe, zeta=zeta,
                                  dc_lam=dc_lam, dc_iters=dc_iters,
                                  pigdm_cg_iters=pigdm_cg_iters, generator=gen,
                                  t_start_guidance=t_guid,
                                  pigdm_sigma_y=pigdm_sigma_y, pigdm_eta=s.get("pigdm_eta", 1e-3),
                                  full=full, t_switch=args.t_switch, noise_model=args.noise_model,
                                  noise_scale=args.noise_scale, dc_schedule=args.dc_schedule,
                                  physics_cfg=cfg["physics"])
            samples.append(xs)
        S = torch.stack(samples)  # [K,B,1,N,N]
        x_hat = S[0]
        x_mean = S.mean(0)
        p_hat = psnr(x_hat, x, op.mask)
        agg["psnr"] += p_hat.tolist()
        agg["ssim"] += ssim(x_hat, x, op.mask).tolist()
        agg["dc"] += data_consistency(op, x_hat, y).tolist()
        agg["psnr_fbp"] += psnr(x_fbp, x, op.mask).tolist()
        agg["ssim_fbp"] += ssim(x_fbp, x, op.mask).tolist()
        if use_lpips:
            agg["lpips"] += lpips_dist(x_hat, x).tolist()
            agg["lpips_fbp"] += lpips_dist(x_fbp, x).tolist()
        # 失败案例：保留 PSNR 最低的 5 张
        for k in range(x.shape[0]):
            worst.append((float(p_hat[k]), idx[k], x[k].cpu(), x_fbp[k].cpu(), x_hat[k].cpu()))
        worst = sorted(worst, key=lambda w: w[0])[:5]
        if n_samples > 1:
            agg["psnr_mean_of_samples"] += psnr(x_mean, x, op.mask).tolist()
            agg["pixel_std"] += S.std(0).flatten(1).mean(1).tolist()
        if first_batch is None:
            first_batch = (x, x_fbp, x_hat, S.std(0) if n_samples > 1 else None)
        print(f"[{i0 + x.shape[0]}/{len(ds)}] psnr={sum(agg['psnr']) / len(agg['psnr']):.2f} "
              f"fbp={sum(agg['psnr_fbp']) / len(agg['psnr_fbp']):.2f}  {(time.time() - t_start) / 60:.1f} min")

    res = {"guidance": args.guidance, "nfe": nfe, "zeta": zeta, "dc_lam": dc_lam, "dc_iters": dc_iters,
           "pigdm_cg_iters": pigdm_cg_iters, "t_start_guidance": t_guid,
           "noise_model": args.noise_model, "noise_scale": args.noise_scale, "dc_schedule": args.dc_schedule,
           **({"t_switch": args.t_switch} if args.guidance == "hybrid" else {}),
           "n_samples": n_samples,
           "split": split, "n_test": len(ds), "ckpt": args.ckpt,
           **({"full_mode": args.full_mode, "kappa": args.kappa, "use_conf": not args.no_conf, "hlcc": args.hlcc,
               "completer": args.completer} if args.guidance == "dcfull" else {}),
           "physics": cfg["physics"], "metrics": summarize(agg), "minutes": (time.time() - t_start) / 60}
    json.dump(res, open(out / "metrics.json", "w"), indent=2)
    # 逐图指标，供按病人聚类的置信区间与配对检验使用
    save_per_image(out, agg)
    save_grid(torch.cat([torch.stack([w[j] for w in worst]) for j in (2, 3, 4)]), out / "failures.png",
              nrow=len(worst))
    yaml.safe_dump(cfg, open(out / "config.yaml", "w"))
    x, x_fbp, x_hat, std = first_batch
    k = min(4, x.shape[0])
    # 前 k 张测试图的 [GT, FBP, 重建, (像素标准差)]，float16，供跨方法定性图使用（所有方法是同一批图）
    arrs = [x[:k], x_fbp[:k], x_hat[:k]] + ([std[:k]] if std is not None else [])
    save_first(out, arrs)
    rows = [x[:k], x_fbp[:k], x_hat[:k]]
    if std is not None:
        rows.append((std[:k] / std[:k].amax(dim=(-1, -2), keepdim=True).clamp(min=1e-6)) * 2 - 1)
    save_grid(torch.cat(rows), out / "samples.png", nrow=k)
    print(json.dumps(res["metrics"], indent=1))
    print("->", out)


if __name__ == "__main__":
    main()
