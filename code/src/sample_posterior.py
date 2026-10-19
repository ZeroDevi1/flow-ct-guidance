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
from src.physics.radon import CTOperator, interp_missing_rows, measured_mask, weighted_cg_dc
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
                     pigdm_sigma_y: float = 0.0, pigdm_eta: float = 1e-3, full: dict | None = None):
    """从 t=0（噪声）积分到 t=1（数据），每步按 guidance 引入测量 y。

    dps  : x ← x + Δt·(v − ζ·∇ₓ‖A x̂₁ − y‖/√s)，s = tr(AᵀA)/N²；ζ 乘 Δt 使总引导量与 NFE 无关，
           除以 √s 使 ζ 与角度数无关。
    pigdm: 解 (AAᵀ + (σ_y²/r_t² + η·s) I) z = y − A x̂₁（s = tr(AAᵀ)/m），
           x ← x + Δt·v + ζ·t₁·∇ₓ⟨x̂₁, Aᵀz⟩；t₁ 对应 ΠGDM 中的信号系数 √ᾱ_t。
    dc   : x̂₁ 做 CG 近端投影 argmin ||A x − y||² + λ||x − x̂₁||²（λ = dc_lam·tr(AᵀA)/N²），
           再按 (ε̂, x̂₁_dc) 重组 x_{t+Δt}（DDNM 式）。
    dcfull: full 为上下文字典 {op, mask, y_full, mode, completer, y_interp, kappa, use_conf, hlcc}；
           x̂₁ 换成 argmin ‖W^{1/2}(A_full x − ŷ)‖² + λ‖x − x̂₁‖²，ŷ 的已测行 = y，未测行由 mode 决定：
           ssm（补全器，权重 κ/σ²_θ，σ² 为补全器预测的方差）、interp（环形插值，权重 κ）、pred（A_full x̂₁，权重 κ）。
    """
    device = y.device
    x = torch.randn(shape, device=device, generator=generator)
    ts = torch.linspace(0, 1, nfe + 1, device=device)
    b = shape[0]
    for i in range(nfe):
        t0, t1 = ts[i], ts[i + 1]
        tb = t0.expand(b)
        dt = t1 - t0
        need_grad = guidance in ("dps", "pigdm") and float(t0) >= t_start_guidance
        with torch.set_grad_enabled(need_grad):
            if need_grad:
                x = x.detach().requires_grad_(True)
            v = model(x, tb)
            x1 = x + (1 - t0) * v
            if guidance == "dps" and need_grad:
                r = op(x1) - y
                norm = r.flatten(1).norm(dim=1)
                g, = torch.autograd.grad(norm.sum(), x)
                # 梯度量级随算子范数 √tr(AᵀA)/N 增长（60 视角约 3.4，180 视角约 5.9），除以它使 ζ 跨视角数可比
                x_next = (x + dt * (v - zeta * g / op.ata_scale ** 0.5)).detach()
            elif guidance == "pigdm" and need_grad:
                r = (y - op(x1)).detach()
                # x_t=(1-t)ε+t x 下 x̂₁ 的后验方差系数 r_t² = (1-t)²/(t²+(1-t)²)（等效 VE σ̃=(1-t)/t）
                r_t2 = float(((1 - t0) ** 2) / (t0 ** 2 + (1 - t0) ** 2 + 1e-8))
                damp = pigdm_sigma_y ** 2 / max(r_t2, 1e-8) + pigdm_eta * op.aat_scale
                # 测量域 CG 解 (A Aᵀ + damp·I) z = r
                z = torch.zeros_like(r); res = r.clone(); p = res.clone()
                rs = (res * res).flatten(1).sum(1)
                for _ in range(pigdm_cg_iters):
                    Ap = op.A_lin(op.A_lin_T(p)) + damp * p
                    alpha = rs / ((p * Ap).flatten(1).sum(1) + 1e-12)
                    z = z + alpha.view(-1, 1, 1, 1) * p
                    res = res - alpha.view(-1, 1, 1, 1) * Ap
                    rs_new = (res * res).flatten(1).sum(1)
                    p = res + (rs_new / (rs + 1e-12)).view(-1, 1, 1, 1) * p
                    rs = rs_new
                w = op.A_lin_T(z).detach()
                g, = torch.autograd.grad((x1 * w).sum(), x)
                x_next = (x + dt * v + zeta * t1 * g).detach()
            else:
                x_next = (x + dt * v).detach()
        if guidance == "dcfull" and float(t0) >= t_start_guidance:
            with torch.no_grad():
                x1 = x1.detach()
                eps_hat = x.detach() - t0 * v.detach()
                x1_dc = _dcfull_step(full, x1, t0, b, dc_lam, dc_iters)
                x_next = (1 - t1) * eps_hat + t1 * x1_dc
        if guidance == "dc" and float(t0) >= t_start_guidance:
            with torch.no_grad():
                x1 = x1.detach()
                eps_hat = x.detach() - t0 * v.detach()
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
            # σ² 以补全器内部的归一化单位计；κ/σ² 截断到 ≤1，未测行的权重不超过实测行
            w_un = (full["kappa"] * torch.exp(-logvar)).clamp(max=1.0)
    elif full["mode"] == "interp":
        y_hat = full["y_interp"]
    elif full["mode"] == "pred":
        y_hat = torch.where(mask.view(1, 1, a, 1), y_full, s_pred)
    else:
        raise ValueError(full["mode"])
    w = torch.where(mask.view(1, a), torch.ones_like(w_un), w_un)
    if full["hlcc"]:
        y_hat = hlcc_project(y_hat, mask, weight=w)
    return weighted_cg_dc(op, x1, y_hat, w, lam=dc_lam, n_iter=dc_iters, relative=True)


def build_full_context(cfg, n, device, y, mode, completer_ckpt=None, kappa=0.3, use_conf=True, use_hlcc=False):
    """为 dcfull 构造上下文：180 角全算子、已测掩码、全网格上的 y、可选补全器 / 插值结果。"""
    p = cfg["physics"]
    mask = measured_mask(180, p.get("n_angles", 180), p.get("angle_range", 3.141592653589793)).to(device)
    op = CTOperator(n, n_angles=180, device=device)
    y_full = y.new_zeros(y.shape[0], 1, 180, y.shape[-1])
    y_full[:, :, mask] = y
    ctx = {"op": op, "mask": mask, "y_full": y_full, "mode": mode, "kappa": kappa, "use_conf": use_conf,
           "hlcc": use_hlcc, "completer": None, "y_interp": None}
    if mode == "interp":
        ctx["y_interp"] = interp_missing_rows(y_full, mask)
    if mode == "ssm":
        ctx["completer"] = completer_ckpt
    return ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--guidance", choices=["dps", "pigdm", "dc", "dcfull", "none"], default="dps")
    ap.add_argument("--full-mode", choices=["ssm", "interp", "pred"], default="ssm", help="dcfull 的未测行来源")
    ap.add_argument("--completer", default=None, help="dcfull/ssm 用的补全器权重（train_completer.py 输出）")
    ap.add_argument("--kappa", type=float, default=0.3, help="dcfull 未测行的基准权重")
    ap.add_argument("--no-conf", action="store_true", help="dcfull/ssm 不用预测置信度，未测行权重固定为 κ")
    ap.add_argument("--hlcc", action="store_true", help="dcfull 中对 ŷ 做 HLCC 投影")
    ap.add_argument("--out", required=True)
    ap.add_argument("--nfe", type=int, default=None)
    ap.add_argument("--zeta", type=float, default=None)
    ap.add_argument("--n-samples", type=int, default=None, help="每张测试图采样次数（不确定性）")
    ap.add_argument("--n-test", type=int, default=None)
    ap.add_argument("--dc-lam", type=float, default=None, help="覆盖 sampling.dc_lam（相对量纲）")
    ap.add_argument("--split", default=None, help="覆盖 data.split；调参用 val，主表用 test200")
    ap.add_argument("--t-start", type=float, default=None, help="覆盖 sampling.t_start_guidance：t 小于该值的步不加引导")
    ap.add_argument("--no-lpips", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    s = cfg["sampling"]
    nfe = args.nfe or s.get("nfe", 50)
    zeta = args.zeta if args.zeta is not None else s.get("zeta", 1.0)
    dc_lam = args.dc_lam if args.dc_lam is not None else s.get("dc_lam", 1.0)
    split = args.split or cfg["data"].get("split", "test")
    t_guid = args.t_start if args.t_start is not None else s.get("t_start_guidance", 0.0)
    n_samples = args.n_samples or s.get("n_samples", 1)
    n_test = args.n_test or cfg["data"].get("n_test", 200)
    bs = s.get("batch_size", 8)
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
                                      use_conf=not args.no_conf, use_hlcc=args.hlcc)
        samples = []
        for _ in range(n_samples):
            xs = posterior_sample(model, op, y, x.shape, args.guidance, nfe, zeta=zeta,
                                  dc_lam=dc_lam, dc_iters=s.get("dc_iters", 5),
                                  pigdm_cg_iters=s.get("pigdm_cg_iters", 5), generator=gen,
                                  t_start_guidance=t_guid,
                                  pigdm_sigma_y=s.get("pigdm_sigma_y", 0.0), pigdm_eta=s.get("pigdm_eta", 1e-3),
                                  full=full)
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

    res = {"guidance": args.guidance, "nfe": nfe, "zeta": zeta, "dc_lam": dc_lam, "t_start_guidance": t_guid,
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
