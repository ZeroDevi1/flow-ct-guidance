"""Track 1 全量按病人聚类 Bootstrap 统计分析与论文图表生成（本地 CPU 运行，零 GPU 消耗）。

用法：
    uv run --with numpy --with matplotlib python code/scripts/analyze_track1.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

N_BOOT = 2000
WIN = (-0.16, 0.24)  # W400/L40 软组织窗


def disk_mask(n: int) -> np.ndarray:
    yy, xx = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n), indexing="ij")
    return (xx ** 2 + yy ** 2) <= 1


def load_exp(d: Path):
    mj = d / "metrics.json"
    if not mj.exists():
        return None
    r = json.loads(mj.read_text())
    rows = list(csv.DictReader(open(d / "per_image.csv"))) if (d / "per_image.csv").exists() else []
    per = {k: np.array([float(x[k]) for x in rows]) for k in (rows[0].keys() if rows else []) if k != "idx"}
    return r, per


def patient_groups(index_csv: Path, n: int) -> np.ndarray:
    pids = [row["patient_id"] for row in csv.DictReader(open(index_csv))][:n]
    uniq = {p: i for i, p in enumerate(dict.fromkeys(pids))}
    return np.array([uniq[p] for p in pids])


def cluster_boot(values: np.ndarray, groups: np.ndarray, rng) -> list[float]:
    g = np.unique(groups)
    sums = np.array([values[groups == k].sum() for k in g])
    cnts = np.array([(groups == k).sum() for k in g])
    idx = rng.integers(0, len(g), size=(N_BOOT, len(g)))
    means = sums[idx].sum(1) / cnts[idx].sum(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def compute_bootstrap(root: Path) -> dict:
    R = root / "results"
    P1B = R / "phase1b"
    T1 = R / "track1"
    groups = patient_groups(R / "index_test200.csv", 200)
    rng = np.random.default_rng(0)

    exps = {
        # Track 1 main test200 experiments
        "main_ld1e5_pigdm_scalar": T1 / "main_ld1e5_pigdm_scalar",
        "main_ld1e5_pigdm_poisson": T1 / "main_ld1e5_pigdm_poisson",
        "main_ld1e5_dc_pwls_const": T1 / "main_ld1e5_dc_pwls_const",
        "main_ld1e5_dc_pwls_rt2": T1 / "main_ld1e5_dc_pwls_rt2",
        "main_ld1e4_pigdm_scalar": T1 / "main_ld1e4_pigdm_scalar",
        "main_ld1e4_pigdm_poisson": T1 / "main_ld1e4_pigdm_poisson",
        "main_ld1e4_dc_pwls_const": T1 / "main_ld1e4_dc_pwls_const",
        "main_ld1e4_dc_pwls_rt2": T1 / "main_ld1e4_dc_pwls_rt2",
        "main_sv30_hybrid_tsw0.4": T1 / "main_sv30_hybrid_tsw0.4",
        "main_sv60_hybrid_tsw0.2": T1 / "main_sv60_hybrid_tsw0.2",
        "main_sv60_hybrid_tsw0.4": T1 / "main_sv60_hybrid_tsw0.4",
        "main_sv60_hybrid_tsw0.6": T1 / "main_sv60_hybrid_tsw0.6",
        "main_sv60_pigdm_bench": T1 / "main_sv60_pigdm_bench",
        "main_sv90_hybrid_tsw0.4": T1 / "main_sv90_hybrid_tsw0.4",
        "main_la120_pigdm_z0.25": T1 / "main_la120_pigdm_z0.25",
        "main_la90_pigdm_z0.25": T1 / "main_la90_pigdm_z0.25",
        "fbpconvnet_la120": T1 / "fbpconvnet_la120",
        "fbpconvnet_la90": T1 / "fbpconvnet_la90",
    }

    summary = {"metrics": {}, "paired": {}}
    loaded = {}
    for name, d in exps.items():
        e = load_exp(d)
        if e is None:
            continue
        r, per = e
        loaded[name] = (r, per)
        m = r["metrics"]
        ci = cluster_boot(per["psnr"], groups[:len(per["psnr"])], rng)
        summary["metrics"][name] = {
            "psnr": m["psnr"]["mean"],
            "ci": ci,
            "ssim": m["ssim"]["mean"],
            "lpips": m["lpips"]["mean"],
            "dc": m["dc"]["mean"],
            "minutes": r.get("minutes", 0.0),
        }

    # 加载参照基线以便计算按病人配对差值
    refs = {
        "p1_sv30_fbpconvnet": R / "fbpconvnet_sv30",
        "p1_sv30_pigdm": R / "main_sv30_pigdm",
        "p1_sv30_dc": R / "main_sv30_dc",
        "p1_sv60_fbpconvnet": R / "fbpconvnet_sv60",
        "p1_sv60_pigdm": R / "main_sv60_pigdm",
        "p1_sv60_dc": R / "main_sv60_dc",
        "p1_sv90_fbpconvnet": R / "fbpconvnet_sv90",
        "p1_sv90_pigdm": R / "main_sv90_pigdm",
        "p1_sv90_dc": R / "main_sv90_dc",
        "p1_ld1e5_fbpconvnet": R / "fbpconvnet_ld1e5",
        "p1_ld1e5_pigdm": R / "main_ld1e5_pigdm",
        "p1_ld1e5_dc": R / "main_ld1e5_dc",
        "p1_ld1e4_pigdm": R / "main_ld1e4_pigdm",
        "p1_ld1e4_dc": R / "main_ld1e4_dc",
        "p1_ld1e4_dps": R / "main_ld1e4_dps",
        "p1b_la120_dc": P1B / "main_la120_dc",
        "p1b_la120_pigdm_old": P1B / "main_la120_pigdm",
        "p1b_la90_dc": P1B / "main_la90_dc",
        "p1b_la90_pigdm_old": P1B / "main_la90_pigdm",
    }
    for k, d in refs.items():
        e = load_exp(d)
        if e is not None:
            loaded[k] = e

    pairs = [
        # T1-1 Low-dose
        ("ld1e5:pigdm_poisson-pigdm", "main_ld1e5_pigdm_poisson", "p1_ld1e5_pigdm"),
        ("ld1e5:pigdm_poisson-pigdm_scalar", "main_ld1e5_pigdm_poisson", "main_ld1e5_pigdm_scalar"),
        ("ld1e5:pigdm_poisson-fbpconvnet", "main_ld1e5_pigdm_poisson", "p1_ld1e5_fbpconvnet"),
        ("ld1e5:pigdm_scalar-pigdm", "main_ld1e5_pigdm_scalar", "p1_ld1e5_pigdm"),
        ("ld1e5:dc_pwls_const-dc", "main_ld1e5_dc_pwls_const", "p1_ld1e5_dc"),
        ("ld1e5:dc_pwls_rt2-dc", "main_ld1e5_dc_pwls_rt2", "p1_ld1e5_dc"),
        ("ld1e4:pigdm_poisson-pigdm", "main_ld1e4_pigdm_poisson", "p1_ld1e4_pigdm"),
        ("ld1e4:pigdm_poisson-pigdm_scalar", "main_ld1e4_pigdm_poisson", "main_ld1e4_pigdm_scalar"),
        ("ld1e4:pigdm_poisson-dps", "main_ld1e4_pigdm_poisson", "p1_ld1e4_dps"),
        ("ld1e4:dc_pwls_rt2-dc", "main_ld1e4_dc_pwls_rt2", "p1_ld1e4_dc"),
        ("ld1e4:dc_pwls_rt2-dc_pwls_const", "main_ld1e4_dc_pwls_rt2", "main_ld1e4_dc_pwls_const"),
        # T1-2 Hybrid
        ("sv30:hybrid0.4-pigdm", "main_sv30_hybrid_tsw0.4", "p1_sv30_pigdm"),
        ("sv30:hybrid0.4-fbpconvnet", "main_sv30_hybrid_tsw0.4", "p1_sv30_fbpconvnet"),
        ("sv60:hybrid0.4-pigdm", "main_sv60_hybrid_tsw0.4", "p1_sv60_pigdm"),
        ("sv60:hybrid0.4-pigdm_bench", "main_sv60_hybrid_tsw0.4", "main_sv60_pigdm_bench"),
        ("sv60:hybrid0.4-dc", "main_sv60_hybrid_tsw0.4", "p1_sv60_dc"),
        ("sv60:hybrid0.4-fbpconvnet", "main_sv60_hybrid_tsw0.4", "p1_sv60_fbpconvnet"),
        ("sv90:hybrid0.4-dc", "main_sv90_hybrid_tsw0.4", "p1_sv90_dc"),
        ("sv90:hybrid0.4-pigdm", "main_sv90_hybrid_tsw0.4", "p1_sv90_pigdm"),
        ("sv90:hybrid0.4-fbpconvnet", "main_sv90_hybrid_tsw0.4", "p1_sv90_fbpconvnet"),
        # T1-2 / T1-3 Limited-angle
        ("la120:pigdm_z0.25-pigdm_z0.5", "main_la120_pigdm_z0.25", "p1b_la120_pigdm_old"),
        ("la120:dc-pigdm_z0.25", "p1b_la120_dc", "main_la120_pigdm_z0.25"),
        ("la120:fbpconvnet-dc", "fbpconvnet_la120", "p1b_la120_dc"),
        ("la90:pigdm_z0.25-pigdm_z0.5", "main_la90_pigdm_z0.25", "p1b_la90_pigdm_old"),
        ("la90:pigdm_z0.25-dc", "main_la90_pigdm_z0.25", "p1b_la90_dc"),
        ("la90:fbpconvnet-pigdm_z0.25", "fbpconvnet_la90", "main_la90_pigdm_z0.25"),
        ("la90:fbpconvnet-dc", "fbpconvnet_la90", "p1b_la90_dc"),
    ]
    for label, a, b in pairs:
        if a in loaded and b in loaded:
            d = loaded[a][1]["psnr"] - loaded[b][1]["psnr"]
            summary["paired"][label] = {
                "diff": float(d.mean()),
                "ci": cluster_boot(d, groups[:len(d)], rng),
            }

    out_p = T1 / "bootstrap_summary.json"
    out_p.write_text(json.dumps(summary, indent=2))
    print("Saved", out_p)
    return summary


def make_track1_solvers_fig(root: Path, summary: dict, out: Path):
    """3 联矢量图：(a) 低剂量噪声感知与零样本迁移；(b) 稀疏视角早 M3 晚 M2 混合调度；(c) 限角 M2 敏感度与 M1 120k 梯度尖峰。"""
    R = root / "results"
    P1B = R / "phase1b"
    T1 = R / "track1"
    M = summary["metrics"]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12.6, 3.65), gridspec_kw={"width_ratios": [1.15, 1.05, 1.05]})

    # ---- Panel (a): Low-dose Poisson awareness (ld1e5 & ld1e4 zero-shot) ----
    labels_a = ["M3\n(naive)", "M3-PWLS\n(const)", "M3-MAP\n($1/r_t^2$)", "M2\n(naive)", "M2-Scalar\n($\\bar\\sigma_y^2$)", "M2-Poisson\n($\\Sigma_y$+PCG)"]
    x = np.arange(len(labels_a))
    w = 0.36
    ld5 = [30.60, M["main_ld1e5_dc_pwls_const"]["psnr"], M["main_ld1e5_dc_pwls_rt2"]["psnr"],
           33.16, M["main_ld1e5_pigdm_scalar"]["psnr"], M["main_ld1e5_pigdm_poisson"]["psnr"]]
    ld4 = [20.63, M["main_ld1e4_dc_pwls_const"]["psnr"], M["main_ld1e4_dc_pwls_rt2"]["psnr"],
           22.53, M["main_ld1e4_pigdm_scalar"]["psnr"], M["main_ld1e4_pigdm_poisson"]["psnr"]]
    b1 = ax1.bar(x - w / 2, ld5, w, label="$I_0{=}10^5$ (tuned)", color="#1f77b4")
    b2 = ax1.bar(x + w / 2, ld4, w, label="$I_0{=}10^4$ (zero-shot)", color="#ff7f0e")
    ax1.axhline(35.32, color="#2ca02c", ls="--", lw=1.2, label="FBPConvNet ($I_0{=}10^5$): 35.32")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.3, f"{h:.1f}", ha="center", va="bottom", fontsize=6.4)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels_a, fontsize=6.9)
    ax1.set_ylabel("PSNR (dB), test200", fontsize=8.5)
    ax1.set_ylim(18, 39.2)
    ax1.set_title("(a) Noise-Aware Solvers & Dose Transfer", fontsize=9.2, fontweight="bold")
    ax1.grid(axis="y", alpha=0.28)
    ax1.legend(fontsize=6.8, loc="upper left", frameon=True)

    # ---- Panel (b): Sparse-view Hybrid (Early M3 -> Late M2) ----
    groups_b = ["30 views", "60 views", "90 views"]
    xb = np.arange(len(groups_b))
    wb = 0.20
    fbp_s = [32.22, 35.50, 37.41]
    m2_s = [32.17, 36.93, 38.09]
    m3_s = [31.62, 35.95, 38.62]
    hyb_s = [M["main_sv30_hybrid_tsw0.4"]["psnr"], M["main_sv60_hybrid_tsw0.4"]["psnr"], M["main_sv90_hybrid_tsw0.4"]["psnr"]]
    for off, vals, lab, col in (
        (-1.5 * wb, fbp_s, "FBPConvNet", "#7f7f7f"),
        (-0.5 * wb, m2_s, "M2 ($\\Pi$GDM)", "#4c78a8"),
        (0.5 * wb, m3_s, "M3 (Prox-DC)", "#f58518"),
        (1.5 * wb, hyb_s, "Hybrid ($t_{\\mathrm{sw}}{=}0.4$)", "#2ca02c"),
    ):
        bars = ax2.bar(xb + off, vals, wb, label=lab, color=col)
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.15, f"{h:.2f}", ha="center", va="bottom", fontsize=5.8, rotation=90)
    ax2.set_xticks(xb)
    ax2.set_xticklabels(groups_b, fontsize=8.2)
    ax2.set_ylabel("PSNR (dB), test200", fontsize=8.5)
    ax2.set_ylim(30.0, 41.6)
    ax2.set_title("(b) Early-M3 / Late-M2 Hybrid Schedule", fontsize=9.2, fontweight="bold")
    ax2.grid(axis="y", alpha=0.28)
    ax2.legend(fontsize=6.8, loc="upper left", frameon=True)

    # ---- Panel (c): Limited-angle M2 sensitivity vs M3 stability (la120 val50) ----
    zetas = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
    m2_val = []
    for z in zetas:
        p_t1 = T1 / f"tune_la120_pigdm_{z:g}" / "metrics.json"
        p_p1b = P1B / f"tune_la120_pigdm_{z:g}" / "metrics.json"
        fp = p_t1 if p_t1.exists() else p_p1b
        m2_val.append(json.loads(fp.read_text())["metrics"]["psnr"]["mean"])
    # M3 val range on la120 across lam in {0.1, 0.3, 1, 3}
    m3_vals = [json.loads((P1B / f"tune_la120_dc_{l:g}" / "metrics.json").read_text())["metrics"]["psnr"]["mean"]
               for l in (0.1, 0.3, 1, 3)]
    ax3.plot(zetas, m2_val, "o-", color="#1f77b4", lw=1.6, ms=5, label="M2 ($\\Pi$GDM) vs. $\\zeta$")
    ax3.axhspan(min(m3_vals), max(m3_vals), color="#f58518", alpha=0.22,
                label=f"M3 across $\\tilde\\lambda\\in[0.1,3]$ ({min(m3_vals):.2f}–{max(m3_vals):.2f})")
    ax3.axhline(max(m3_vals), color="#f58518", ls="--", lw=1.2)
    ax3.annotate(f"peak $\\zeta{{=}}0.25$\n({m2_val[2]:.2f} dB)", xy=(0.25, m2_val[2]), xytext=(0.45, 29.6),
                 arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=0.9), fontsize=7.0, color="#1f77b4")
    ax3.set_xscale("log")
    ax3.set_xticks(zetas)
    ax3.set_xticklabels([f"{z:g}" for z in zetas], fontsize=7.5)
    ax3.set_xlabel("M2 guidance strength $\\zeta$ ($120^\\circ$ lim., val50)", fontsize=8.2)
    ax3.set_ylabel("val PSNR (dB)", fontsize=8.5)
    ax3.set_ylim(10.5, 32.0)
    ax3.set_title("(c) $120^\\circ$ Sensitivity: M2 Cliff vs. M3 Plateau", fontsize=9.2, fontweight="bold")
    ax3.grid(alpha=0.28)
    ax3.legend(fontsize=6.7, loc="lower left", frameon=True)

    plt.tight_layout(pad=0.8)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out)


def make_m1_grad_fig(root: Path, out: Path):
    """120k 终版权重下的 M1 引导梯度尖峰曲线 + t_start=0 验证对照。"""
    d = json.loads((root / "results/track1/m1_grad_120k.json").read_text())
    u_steps = d["unguided"]["steps"]
    g_steps = d["guided_ts04_z30"]["steps"]
    ts = [s["t"] for s in u_steps]
    u_rms = [s["grad_rms"] for s in u_steps]
    g_rms = [s["grad_rms"] for s in g_steps]
    u_vel = [s["vel_rms"] for s in u_steps]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.8, 3.1))
    ax1.plot(ts, u_rms, "o-", ms=2.6, lw=1.4, color="#d62728", label="M1 $\\mathrm{RMS}(g_t)$ (unguided traj.)")
    ax1.plot(ts, g_rms, "s--", ms=2.2, lw=1.2, color="#1f77b4", alpha=0.85, label="M1 $\\mathrm{RMS}(g_t)$ ($t_s{=}0.4, \\zeta{=}30$)")
    ax1.plot(ts, u_vel, ":", lw=1.3, color="#555555", label="Velocity $\\mathrm{RMS}(v_\\theta)$")
    ax1.set_yscale("log")
    ax1.set_xlabel("Flow time $t$", fontsize=8.8)
    ax1.set_ylabel("RMS magnitude (log scale)", fontsize=8.8)
    ax1.set_title("(a) 120k Final Prior: M1 Gradient Spike at $t{\\approx}0$", fontsize=9.2)
    ax1.annotate(f"$t{{=}}0$: {u_rms[0]:.2f}\n({d['summary_unguided']['spike_ratio_vs_mean']:.1f}$\\times$ late mean)",
                 xy=(0.0, u_rms[0]), xytext=(0.18, 0.48),
                 arrowprops=dict(arrowstyle="->", color="#d62728", lw=0.9), fontsize=7.5, color="#d62728")
    ax1.grid(alpha=0.28)
    ax1.legend(fontsize=7.2, loc="lower right")

    ts0 = d["ts0_val_psnr_grid"]
    zs = [float(k) for k in ts0.keys()]
    ps = [ts0[k] for k in ts0.keys()]
    ax2.plot(zs, ps, "o-", lw=1.5, ms=5, color="#d62728", label="M1 from $t_s{=}0.0$ (120k final)")
    ax2.axhline(24.59, color="#1f77b4", ls="--", lw=1.2, label="M1 best $t_s{=}0.4, \\zeta{=}30$ (24.59 dB)")
    ax2.axhline(37.67, color="#2ca02c", ls=":", lw=1.3, label="Hybrid $t_{\\mathrm{sw}}{=}0.4$ (37.67 dB)")
    ax2.set_xscale("log")
    ax2.set_xticks(zs)
    ax2.set_xticklabels([f"{z:g}" for z in zs])
    ax2.set_xlabel("M1 strength $\\zeta$ (60 views, val50)", fontsize=8.8)
    ax2.set_ylabel("val PSNR (dB)", fontsize=8.8)
    ax2.set_ylim(13, 39.5)
    ax2.set_title("(b) Starting M1 at $t_s{=}0$ Fails Even on 120k Prior", fontsize=9.2)
    ax2.grid(alpha=0.28)
    ax2.legend(fontsize=7.2, loc="center right")

    plt.tight_layout(pad=0.8)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out)


def make_track1_qualitative_fig(root: Path, out: Path, img: int = 3):
    """低剂量 ld1e5 / ld1e4（零样本）与稀疏 60 视角的新求解器定性对比 + 误差图（W400/L40）。"""
    R = root / "results"
    T1 = R / "track1"
    rows_spec = [
        ("LD $I_0{=}10^5$", [
            ("GT", T1 / "main_ld1e5_pigdm_poisson", 0),
            ("FBP", T1 / "main_ld1e5_pigdm_poisson", 1),
            ("M3-PWLS", T1 / "main_ld1e5_dc_pwls_const", 2),
            ("M3-MAP", T1 / "main_ld1e5_dc_pwls_rt2", 2),
            ("M2-Scalar", T1 / "main_ld1e5_pigdm_scalar", 2),
            ("M2-Poisson", T1 / "main_ld1e5_pigdm_poisson", 2),
        ]),
        ("LD $I_0{=}10^4$\n(zero-shot)", [
            ("GT", T1 / "main_ld1e4_pigdm_poisson", 0),
            ("FBP", T1 / "main_ld1e4_pigdm_poisson", 1),
            ("M3-PWLS", T1 / "main_ld1e4_dc_pwls_const", 2),
            ("M3-MAP", T1 / "main_ld1e4_dc_pwls_rt2", 2),
            ("M2-Scalar", T1 / "main_ld1e4_pigdm_scalar", 2),
            ("M2-Poisson", T1 / "main_ld1e4_pigdm_poisson", 2),
        ]),
        ("Sparse 60v", [
            ("GT", T1 / "main_sv60_hybrid_tsw0.4", 0),
            ("FBP", T1 / "main_sv60_hybrid_tsw0.4", 1),
            ("M2 (ΠGDM)", T1 / "main_sv60_pigdm_bench", 2),
            ("Hybrid 0.2", T1 / "main_sv60_hybrid_tsw0.2", 2),
            ("Hybrid 0.6", T1 / "main_sv60_hybrid_tsw0.6", 2),
            ("Hybrid 0.4", T1 / "main_sv60_hybrid_tsw0.4", 2),
        ]),
    ]
    ncols = 6
    fig, axes = plt.subplots(6, ncols, figsize=(2.05 * ncols, 2.0 * 6))
    for block_idx, (row_label, cols) in enumerate(rows_spec):
        r_img, r_err = block_idx * 2, block_idx * 2 + 1
        ref = np.load(cols[0][1] / "first4.npy")[0, img].astype(np.float32)
        m = disk_mask(ref.shape[-1])
        for j, (cname, folder, slot) in enumerate(cols):
            ax_i, ax_e = axes[r_img, j], axes[r_err, j]
            ax_i.axis("off"); ax_e.axis("off")
            arr = np.load(folder / "first4.npy")[slot, img].astype(np.float32)
            ax_i.imshow(np.where(m, arr, -1.0), cmap="gray", vmin=WIN[0], vmax=WIN[1])
            ax_i.set_title(cname, fontsize=8.5, pad=3)
            if j == 0:
                ax_e.imshow(np.zeros_like(ref), cmap="magma", vmin=0, vmax=0.18)
                ax_e.text(128, 128, "Reference\n(|err| = 0)", color="white", ha="center", va="center", fontsize=8)
            else:
                mse = float(np.mean((arr - ref)[m] ** 2))
                p = 10.0 * np.log10(4.0 / max(mse, 1e-12))
                ax_i.text(5, 248, f"{p:.1f} dB", color="yellow", fontsize=7.5,
                          bbox=dict(facecolor="black", alpha=0.45, pad=1.2, edgecolor="none"))
                ax_e.imshow(np.where(m, np.abs(arr - ref), 0.0), cmap="magma", vmin=0, vmax=0.18)
        axes[r_img, 0].text(-18, 128, row_label, rotation=90, va="center", ha="right", fontsize=9, fontweight="bold")
        axes[r_err, 0].text(-18, 128, "|Error|", rotation=90, va="center", ha="right", fontsize=8.5)
    plt.tight_layout(pad=0.6)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out)


def update_limited_angle_fig(root: Path, out: Path, img: int = 3):
    """更新限角定性对比图：加入补齐后的 FBPConvNet 监督基线与重标定 M2 (zeta=0.25)。"""
    P1B = root / "results/phase1b"
    T1 = root / "results/track1"
    rows_spec = [
        ("120° Limited", [
            ("GT", P1B / "fbp_la120", 0),
            ("FBP", P1B / "fbp_la120", 1),
            ("FBPConvNet", T1 / "fbpconvnet_la120", 2),
            ("M2 (ζ=0.5)", P1B / "main_la120_pigdm", 2),
            ("M2 (ζ=0.25)", T1 / "main_la120_pigdm_z0.25", 2),
            ("M3 (Prox-DC)", P1B / "main_la120_dc", 2),
        ]),
        ("90° Limited", [
            ("GT", P1B / "fbp_la90", 0),
            ("FBP", P1B / "fbp_la90", 1),
            ("FBPConvNet", T1 / "fbpconvnet_la90", 2),
            ("M2 (ζ=0.5)", P1B / "main_la90_pigdm", 2),
            ("M2 (ζ=0.25)", T1 / "main_la90_pigdm_z0.25", 2),
            ("M3 (Prox-DC)", P1B / "main_la90_dc", 2),
        ]),
    ]
    ncols = 6
    fig, axes = plt.subplots(4, ncols, figsize=(2.05 * ncols, 2.05 * 4))
    for block_idx, (row_label, cols) in enumerate(rows_spec):
        r_img, r_err = block_idx * 2, block_idx * 2 + 1
        ref = np.load(cols[0][1] / "first4.npy")[0, img].astype(np.float32)
        m = disk_mask(ref.shape[-1])
        for j, (cname, folder, slot) in enumerate(cols):
            ax_i, ax_e = axes[r_img, j], axes[r_err, j]
            ax_i.axis("off"); ax_e.axis("off")
            arr = np.load(folder / "first4.npy")[slot, img].astype(np.float32)
            ax_i.imshow(np.where(m, arr, -1.0), cmap="gray", vmin=WIN[0], vmax=WIN[1])
            ax_i.set_title(cname, fontsize=8.5, pad=3)
            if j == 0:
                ax_e.imshow(np.zeros_like(ref), cmap="magma", vmin=0, vmax=0.25)
                ax_e.text(128, 128, "Reference\n(|err| = 0)", color="white", ha="center", va="center", fontsize=8)
            else:
                mse = float(np.mean((arr - ref)[m] ** 2))
                p = 10.0 * np.log10(4.0 / max(mse, 1e-12))
                ax_i.text(5, 248, f"{p:.1f} dB", color="yellow", fontsize=7.5,
                          bbox=dict(facecolor="black", alpha=0.45, pad=1.2, edgecolor="none"))
                ax_e.imshow(np.where(m, np.abs(arr - ref), 0.0), cmap="magma", vmin=0, vmax=0.25)
        axes[r_img, 0].text(-18, 128, row_label, rotation=90, va="center", ha="right", fontsize=9, fontweight="bold")
        axes[r_err, 0].text(-18, 128, "|Error|", rotation=90, va="center", ha="right", fontsize=8.5)
    plt.tight_layout(pad=0.6)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out)


def main():
    root = Path(__file__).resolve().parents[2]
    summary = compute_bootstrap(root)
    fig_dir = root / "paper/figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    make_track1_solvers_fig(root, summary, fig_dir / "track1_solvers.pdf")
    make_m1_grad_fig(root, fig_dir / "m1_grad_120k.pdf")
    make_track1_qualitative_fig(root, fig_dir / "track1_qualitative.png")
    update_limited_angle_fig(root, fig_dir / "limited_angle.png")
    print("\n=== TRACK 1 BOOTSTRAP SUMMARY ===")
    for k, v in summary["metrics"].items():
        print(f"{k:28s} PSNR={v['psnr']:6.2f} [{v['ci'][0]:5.2f}, {v['ci'][1]:5.2f}]  "
              f"SSIM={v['ssim']:.3f}  LPIPS={v['lpips']:.3f}  DC={v['dc']:.4f}  min={v['minutes']:.2f}")
    print("\n=== PAIRED DIFFERENCES (patient-clustered 95% CI) ===")
    for k, v in summary["paired"].items():
        print(f"{k:36s} diff={v['diff']:+6.2f} dB  [{v['ci'][0]:+6.2f}, {v['ci'][1]:+6.2f}]")


if __name__ == "__main__":
    main()
