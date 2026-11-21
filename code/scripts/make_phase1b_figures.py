"""生成 Phase 1b 论文补充图表（本地 CPU 运行，零 GPU 消耗）：
1. paper/figures/limited_angle.png  —— 限角 120° / 90° 的重建定性对比与绝对误差图
2. paper/figures/phase1b_probe.pdf  —— 1D SSM 补全器验证误差收敛曲线（对比 2D 先验投影）+ 定量柱状图
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WIN = (-0.16, 0.24)  # W400/L40 软组织窗，与 analyze.py 保持一致


def disk_mask(n: int) -> np.ndarray:
    yy, xx = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n), indexing="ij")
    return (xx ** 2 + yy ** 2) <= 1


def make_limited_angle_fig(p1b: Path, out: Path, img: int = 3):
    rows_spec = [
        ("120° Limited", "la120", [
            ("GT", "fbp_la120", 0),
            ("FBP", "fbp_la120", 1),
            ("dcfull-interp", "main_la120_dcfull_interp", 2),
            ("M2 (ΠGDM)", "main_la120_pigdm", 2),
            ("M3 (Prox-DC)", "main_la120_dc", 2),
            ("SSM+HLCC", "main_la120_dcfull_ssm_hlcc", 2),
        ]),
        ("90° Limited", "la90", [
            ("GT", "fbp_la90", 0),
            ("FBP", "fbp_la90", 1),
            ("dcfull-interp", "main_la90_dcfull_interp", 2),
            ("M2 (ΠGDM)", "main_la90_pigdm", 2),
            ("M3 (Prox-DC)", "main_la90_dc", 2),
            ("dcfull-pred", "main_la90_dcfull_pred", 2),
        ]),
    ]
    ncols = 6
    fig, axes = plt.subplots(4, ncols, figsize=(2.05 * ncols, 2.05 * 4))
    for block_idx, (row_label, _, cols) in enumerate(rows_spec):
        r_img = block_idx * 2
        r_err = block_idx * 2 + 1
        ref = np.load(p1b / cols[0][1] / "first4.npy")[0, img].astype(np.float32)
        m = disk_mask(ref.shape[-1])
        for j, (cname, folder, slot) in enumerate(cols):
            ax_i = axes[r_img, j]
            ax_e = axes[r_err, j]
            ax_i.axis("off"); ax_e.axis("off")
            arr = np.load(p1b / folder / "first4.npy")[slot, img].astype(np.float32)
            ax_i.imshow(np.where(m, arr, -1.0), cmap="gray", vmin=WIN[0], vmax=WIN[1])
            ax_i.set_title(cname, fontsize=8.5, pad=3)
            if j == 0:
                ax_e.imshow(np.zeros_like(ref), cmap="magma", vmin=0, vmax=0.25)
                ax_e.text(128, 128, "Reference\n(|err| = 0)", color="white",
                          ha="center", va="center", fontsize=8)
            else:
                mse = float(np.mean((arr - ref)[m] ** 2))
                psnr = 10.0 * np.log10(4.0 / max(mse, 1e-12))
                ax_i.text(5, 248, f"{psnr:.1f} dB", color="yellow", fontsize=7.5,
                          bbox=dict(facecolor="black", alpha=0.45, pad=1.2, edgecolor="none"))
                err = np.where(m, np.abs(arr - ref), 0.0)
                ax_e.imshow(err, cmap="magma", vmin=0, vmax=0.25)
        axes[r_img, 0].text(-18, 128, row_label, rotation=90, va="center", ha="right",
                            fontsize=9, fontweight="bold")
        axes[r_err, 0].text(-18, 128, "|Error|", rotation=90, va="center", ha="right", fontsize=8.5)
    plt.tight_layout(pad=0.6)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out)


def make_probe_fig(p1b: Path, val_log_path: Path, out: Path):
    vlog = json.loads(val_log_path.read_text())
    steps = [r["step"] / 1000.0 for r in vlog]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.4), gridspec_kw={"width_ratios": [1.15, 1.0]})

    # 左图：验证集正弦图相对误差收敛曲线（completer vs prior_pred 水平基线）
    palette = {"sv60": "#1f77b4", "sv30": "#2ca02c", "la120": "#d62728", "la90": "#9467bd"}
    labels = {"sv60": "60 views", "sv30": "30 views", "la120": "120° lim.", "la90": "90° lim."}
    for k in ("sv60", "sv30", "la120", "la90"):
        ys = [r[k]["completer"] * 100.0 for r in vlog]
        ref = vlog[-1][k]["prior_pred"] * 100.0
        ax1.plot(steps, ys, "o-", ms=3.2, lw=1.4, color=palette[k], label=f"{labels[k]} (SSM)")
        ax1.axhline(ref, ls="--", lw=1.1, color=palette[k], alpha=0.75)
    ax1.plot([], [], "k--", lw=1.1, label="2D prior $A_{\\mathrm{full}}(\\hat{x}_1)$")
    ax1.set_xlabel("Training step ($\\times 10^3$)", fontsize=9)
    ax1.set_ylabel("Unmeasured sinogram rel. error (%)", fontsize=9)
    ax1.set_title("(a) 1D SSM vs. 2D Prior Projection (val, $t{=}0.8$)", fontsize=9.5)
    ax1.set_ylim(0.25, 1.78)
    ax1.grid(alpha=0.28)
    ax1.legend(fontsize=7.2, ncol=2, loc="upper right", frameon=True)

    # 右图：限角（la120 / la90）与稀疏（sv60）关键方法对比
    groups = ["60 views", "120° lim.", "90° lim."]
    x = np.arange(len(groups))
    w = 0.20
    bs = json.loads((p1b / "bootstrap_summary.json").read_text())
    m2 = [36.93, bs["main_la120_pigdm"]["psnr"], bs["main_la90_pigdm"]["psnr"]]
    m3 = [35.95, bs["main_la120_dc"]["psnr"], bs["main_la90_dc"]["psnr"]]
    interp = [bs["main_sv60_dcfull_interp"]["psnr"], bs["main_la120_dcfull_interp"]["psnr"], bs["main_la90_dcfull_interp"]["psnr"]]
    ssm = [bs["main_sv60_dcfull_ssm_hlcc"]["psnr"], bs["main_la120_dcfull_ssm_hlcc"]["psnr"], np.nan]

    b1 = ax2.bar(x - 1.5 * w, m2, w, label="M2 (ΠGDM)", color="#4c78a8")
    b2 = ax2.bar(x - 0.5 * w, m3, w, label="M3 (Prox-DC)", color="#f58518")
    b3 = ax2.bar(x + 0.5 * w, interp, w, label="dcfull-interp", color="#bab0ac")
    b4 = ax2.bar(x[:2] + 1.5 * w, ssm[:2], w, label="dcfull-SSM+HLCC", color="#54a24b")

    for bars in (b1, b2, b3, b4):
        for bar in bars:
            h = bar.get_height()
            if np.isfinite(h):
                ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.25, f"{h:.1f}",
                         ha="center", va="bottom", fontsize=6.6)
    ax2.set_xticks(x)
    ax2.set_xticklabels(groups, fontsize=8.5)
    ax2.set_ylabel("PSNR (dB), test200", fontsize=9)
    ax2.set_ylim(16, 40.5)
    ax2.set_title("(b) Geometry Flip (M2 vs. M3) & Completion Probe", fontsize=9.5)
    ax2.grid(axis="y", alpha=0.28)
    ax2.legend(fontsize=7.2, loc="upper right", frameon=True)

    plt.tight_layout(pad=0.8)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out)


def make_architecture_fig(out: Path):
    """矢量模型架构与采样注入总览图：(a) 19.8M 共享 2D U-Net + ODE 注入规则；(b) 2.57M 双向角度 SSM 补全器。"""
    from matplotlib.patches import FancyBboxPatch

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.2, 4.35), gridspec_kw={"width_ratios": [1.28, 1.0]}
    )
    for ax in (ax1, ax2):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")

    def box(ax, x, y, w, h, text, fc="#eef4f9", ec="#184e77", fs=7.3, bold=False, tc="#111111"):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.14",
                           facecolor=fc, edgecolor=ec, linewidth=1.05)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", color=tc, linespacing=1.18)

    def arrow(ax, x1, y1, x2, y2, color="#333333", lw=1.0, ls="-"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls, shrinkA=1, shrinkB=1))

    # ---- Panel (a): 19.8M Shared 2D U-Net + ODE Injection ----
    ax1.text(5.0, 9.65, "(a) Shared 19.8M 2D Rectified-Flow U-Net & ODE Measurement Injection",
             ha="center", va="center", fontsize=9.2, fontweight="bold", color="#184e77")

    # Input & Time embedding
    box(ax1, 0.25, 8.25, 2.0, 0.95, "$x_t \\in \\mathbb{R}^{1\\times 256^2}$\n(FBP for FBPConvNet)", fc="#f7f7f7", ec="#555555", fs=7.0)
    box(ax1, 3.35, 8.35, 3.3, 0.85, "Sinusoidal $t \\to$ MLP (256d)\nAdaptive GroupNorm FiLM", fc="#fff3e0", ec="#e67e22", fs=7.0)
    box(ax1, 7.65, 8.25, 2.1, 0.95, "$v_\\theta(x_t,t) \\to \\hat{x}_1, \\hat{\\epsilon}$\n(residual if $t\\equiv 0$)", fc="#f7f7f7", ec="#555555", fs=7.0)

    # 5-stage U-Net encoder & decoder
    enc_specs = [
        ("256² · 64ch", 7.10),
        ("128² · 64ch", 6.15),
        ("64² · 128ch", 5.20),
        ("32² · 128ch", 4.25),
        ("16² · 256ch + Attn", 3.25),
    ]
    for i, (lab, y) in enumerate(enc_specs):
        fc = "#d9eaf7" if i < 4 else "#b8d8f0"
        box(ax1, 0.35, y, 2.55, 0.68, f"Enc {i+1}: 2×Res\n{lab}", fc=fc, ec="#184e77", fs=6.7)
        box(ax1, 7.10, y, 2.55, 0.68, f"Dec {i+1}: 3×Res\n{lab}", fc=fc, ec="#184e77", fs=6.7)
        # Skip connection dashed arrow (levels 1..4; level 5 routes via bottleneck)
        if i < 4:
            arrow(ax1, 2.95, y + 0.34, 7.05, y + 0.34, color="#6c8ebf", lw=0.85, ls="--")
            arrow(ax1, 1.62, y, 1.62, enc_specs[i + 1][1] + 0.70, color="#184e77", lw=0.9)
            arrow(ax1, 8.38, enc_specs[i + 1][1] + 0.70, 8.38, y, color="#184e77", lw=0.9)
    ax1.text(5.0, 7.65, "skip connections", ha="center", va="center", fontsize=6.4, color="#52759c", style="italic")

    # Bottleneck
    box(ax1, 3.45, 3.15, 3.10, 0.88, "Bottleneck (16² · 256ch)\nRes → 4-Head Self-Attn → Res",
        fc="#cfe2f3", ec="#184e77", fs=6.8, bold=True)
    arrow(ax1, 2.95, 3.59, 3.42, 3.59, color="#184e77", lw=1.0)
    arrow(ax1, 6.58, 3.59, 7.05, 3.59, color="#184e77", lw=1.0)
    arrow(ax1, 1.25, 8.23, 1.25, 7.82, color="#333333", lw=1.0)
    arrow(ax1, 8.65, 7.82, 8.65, 8.23, color="#333333", lw=1.0)
    arrow(ax1, 5.0, 8.33, 5.0, 4.08, color="#e67e22", lw=0.85, ls=":")

    # ODE Injection Strip at bottom
    box(ax1, 0.25, 0.35, 9.50, 2.25, "", fc="#f8fafc", ec="#4a6fa5", fs=7.0)
    ax1.text(5.0, 2.34, "Euler ODE Step ($t \\to t_+$) — Four Controlled Measurement-Injection Rules",
             ha="center", va="center", fontsize=7.6, fontweight="bold", color="#184e77")
    rules = [
        ("M1: DPS-grad\n$-\\Delta t\\frac{\\zeta}{\\sqrt{s_x}}\\nabla\\|A\\hat{x}_1-y\\|$\n(U-Net backprop)", 0.42, "#fdecea", "#c0392b"),
        ("M2: ΠGDM\n5-CG $(AA^\\top+\\eta s_y I)^{-1}r$\n+ VJP through U-Net", 2.74, "#e8f4fd", "#1f77b4"),
        ("M3: Prox-DC\n5-CG image-space DC\n(no U-Net backprop)", 5.06, "#eaf7ea", "#2e7d32"),
        ("M3-Full (dcfull)\nFull $180^\\circ$ weighted CG\nusing Panel (b) + HLCC", 7.38, "#f3ecfa", "#6a3d9a"),
    ]
    for txt, rx, rfc, rec in rules:
        box(ax1, rx, 0.55, 2.18, 1.50, txt, fc=rfc, ec=rec, fs=6.3)

    # ---- Panel (b): 2.57M Bidirectional Angle-Sequence SSM ----
    ax2.text(5.0, 9.65, "(b) 2.57M Bidirectional Angle-Sequence SSM (Phase 1b)",
             ha="center", va="center", fontsize=9.2, fontweight="bold", color="#6a3d9a")

    stages_b = [
        (8.35, 0.90, "Inputs (RMS-normalized by measured rows)\n$[s_{\\mathrm{pred}},\\ y\\odot m,\\ m]\\in\\mathbb{R}^{3\\times 180\\times 256}$  +  time $t$", "#f7f7f7", "#555555"),
        (7.05, 0.82, "$2\\pi$ Periodic Extension: $s(\\theta{+}π,r)=s(\\theta,-r)$\n180 angles $\\to$ 360 strictly periodic rows", "#f3ecfa", "#6a3d9a"),
        (5.70, 0.88, "Detector 1D Conv Encoder ($r$: $256\\to 64$, 16ch)\nFlatten $\\to$ Linear(1024$\\to$256) + Cyclic PosEnc ($\\sin/\\cos\\theta,2\\theta$)", "#ede7f6", "#6a3d9a"),
        (3.85, 1.35, "4× Bidirectional Selective SSM Blocks (Fused Triton Scan)\nTwo-lap ring scan ($L=720$, take 2nd lap)\nLayerNorm $\\to$ $t$-FiLM $\\to$ Bi-SSM ($d{=}256, d_i{=}512, N{=}16$)", "#e1d5f0", "#512da8"),
        (2.40, 0.95, "Zero-Init Linear Decoder (first 180 rows)\nResidual $\\delta s \\in \\mathbb{R}^{180\\times 256}$  &  Log-Var $\\log\\sigma_\\theta^2 \\in [-10,4]$", "#ede7f6", "#6a3d9a"),
        (0.55, 1.30, "Clamp measured rows $\\hat{y}_m = y$ ($w_m=1$)\nNormalized weights $w_\\theta = \\min(1, \\kappa\\bar{\\sigma}^2/\\sigma_\\theta^2)$\nOptional HLCC Projection $\\Pi_{\\mathrm{HLCC}}(\\hat{y};w)$ $\\to$ Weighted CG", "#eaf7ea", "#2e7d32"),
    ]
    for idx, (y, h, txt, fc, ec) in enumerate(stages_b):
        box(ax2, 0.45, y, 9.10, h, txt, fc=fc, ec=ec, fs=6.9, bold=(idx == 3))
        if idx < len(stages_b) - 1:
            ny = stages_b[idx + 1][0] + stages_b[idx + 1][1]
            arrow(ax2, 5.0, y, 5.0, ny + 0.02, color="#512da8", lw=1.05)

    plt.tight_layout(pad=0.6)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out)


def main():
    root = Path(__file__).resolve().parents[2]
    p1b = root / "results" / "phase1b"
    fig_dir = root / "paper" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    make_architecture_fig(fig_dir / "architecture.pdf")
    make_limited_angle_fig(p1b, fig_dir / "limited_angle.png")
    make_probe_fig(p1b, root / "code" / "checkpoints" / "completer_ssm" / "val_log.json",
                   fig_dir / "phase1b_probe.pdf")


if __name__ == "__main__":
    main()
