"""论文表格与图：读取 results/ 下各实验的 metrics.json / per_image.csv / first4.npy，输出 LaTeX 表与 PDF/PNG 图。

用法（本地，无需 GPU）：
    uv run --with numpy --with matplotlib python scripts/analyze.py \
        --results results --index results/index_test200.csv --paper ../papers/arxiv_phase1

置信区间：test200 是 20 个病人 × 10 张，切片间相关，按病人整簇重采样（bootstrap 2000 次）给 95% CI；
配对差值同样按病人重采样，用于"方法 A 是否优于 B"的判断。
缺失的实验自动跳过（表中记为 --），便于在矩阵未跑完时预览。
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

SETTINGS = [("sv30", "30 views"), ("sv60", "60 views"), ("sv90", "90 views"), ("ld1e5", r"LD $I_0{=}10^5$")]
GEN = [("dps", "M1 DPS-grad"), ("pigdm", r"M2 $\Pi$GDM"), ("dc", "M3 Prox-DC")]
N_BOOT = 2000


def load_exp(d: Path):
    if not (d / "metrics.json").exists():
        return None
    r = json.load(open(d / "metrics.json"))
    rows = list(csv.DictReader(open(d / "per_image.csv"))) if (d / "per_image.csv").exists() else []
    per = {k: np.array([float(x[k]) for x in rows]) for k in (rows[0].keys() if rows else []) if k != "idx"}
    return r, per


def patient_groups(index_csv: Path, n: int) -> np.ndarray:
    """返回每张切片所属病人的整数编号；没有索引文件时退化为每张图各自成簇。"""
    if index_csv and index_csv.exists():
        pids = [row["patient_id"] for row in csv.DictReader(open(index_csv))][:n]
        uniq = {p: i for i, p in enumerate(dict.fromkeys(pids))}
        return np.array([uniq[p] for p in pids])
    return np.arange(n)


def cluster_boot(values: np.ndarray, groups: np.ndarray, rng) -> tuple[float, float]:
    """按病人整簇重采样均值的 95% 置信区间。"""
    g = np.unique(groups)
    sums = np.array([values[groups == k].sum() for k in g]); cnts = np.array([(groups == k).sum() for k in g])
    idx = rng.integers(0, len(g), size=(N_BOOT, len(g)))
    means = sums[idx].sum(1) / cnts[idx].sum(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def fmt_ci(mean, ci, nd=2):
    return f"{mean:.{nd}f}" + (rf"\,{{\scriptsize[{ci[0]:.{nd}f},{ci[1]:.{nd}f}]}}" if ci else "")


def main_table(R: Path, groups, rng, out: Path):
    lines = [r"\begin{tabular}{llcccc}", r"\toprule",
             r"Setting & Method & PSNR$\uparrow$ [95\% CI] & SSIM$\uparrow$ & LPIPS$\downarrow$ & DC$\downarrow$ \\", r"\midrule"]
    summary = {}
    for s, sname in SETTINGS:
        rows = [("FBP", R / f"fbp_{s}"), ("FBPConvNet (sup.)", R / f"fbpconvnet_{s}")] + \
               [(name, R / f"main_{s}_{m}") for m, name in GEN]
        best = None
        cells = []
        for name, d in rows:
            e = load_exp(d)
            if e is None:
                cells.append((name, None)); continue
            r, per = e
            m = r["metrics"]
            ci = cluster_boot(per["psnr"], groups[:len(per["psnr"])], rng) if "psnr" in per else None
            cells.append((name, (m["psnr"]["mean"], ci, m["ssim"]["mean"], m.get("lpips", {}).get("mean"), m["dc"]["mean"])))
            summary[(s, name)] = m["psnr"]["mean"]
            if best is None or m["psnr"]["mean"] > best:
                best = m["psnr"]["mean"]
        for i, (name, v) in enumerate(cells):
            lab = rf"\multirow{{{len(cells)}}}{{*}}{{{sname}}}" if i == 0 else ""
            if v is None:
                lines.append(f"{lab} & {name} & -- & -- & -- & -- \\\\"); continue
            p, ci, ss, lp, dc = v
            ptxt = fmt_ci(p, ci)
            if abs(p - best) < 1e-9:
                ptxt = r"\textbf{" + f"{p:.2f}" + "}" + ptxt[len(f"{p:.2f}"):]
            lines.append(f"{lab} & {name} & {ptxt} & {ss:.3f} & {'--' if lp is None else f'{lp:.3f}'} & {dc:.4f} \\\\")
        lines.append(r"\midrule" if s != SETTINGS[-1][0] else r"\bottomrule")
    lines.append(r"\end{tabular}")
    out.write_text("\n".join(lines) + "\n")
    return summary


def paired(R: Path, a: str, b: str, groups, rng, col_a: str = "psnr", col_b: str = "psnr"):
    """按病人重采样配对差 a[col_a] − b[col_b] 的均值与 95% CI（默认比较单样本 PSNR）。"""
    ea, eb = load_exp(R / a), load_exp(R / b)
    if ea is None or eb is None or col_a not in ea[1] or col_b not in eb[1]:
        return None
    d = ea[1][col_a] - eb[1][col_b]
    return float(d.mean()), cluster_boot(d, groups[:len(d)], rng)


def disk_mask(n: int) -> np.ndarray:
    """重建域（内切圆）掩码，与 radon.circle_mask 一致；圆外不受测量约束，显示与指标都只看圆内。"""
    yy, xx = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n), indexing="ij")
    return (xx ** 2 + yy ** 2) <= 1


# 软组织窗 W400/L40：HU [-160, 240] → 归一化域 [-0.16, 0.24]
WIN = (-0.16, 0.24)


def fig_qualitative(R: Path, out: Path, img: int = 3):
    """img=3：test200 第 4 张（心脏、双肺、主动脉、椎体），比前三张腹部切片结构更丰富。"""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = [("GT", None), ("FBP", None), ("FBPConvNet", "fbpconvnet_{s}"), ("M1", "main_{s}_dps"),
            ("M2", "main_{s}_pigdm"), ("M3", "main_{s}_dc")]
    fig, axes = plt.subplots(len(SETTINGS), len(cols), figsize=(2.0 * len(cols), 2.0 * len(SETTINGS)))
    for i, (s, sname) in enumerate(SETTINGS):
        ref = None
        for j, (cname, pat) in enumerate(cols):
            ax = axes[i, j]; ax.axis("off")
            if pat is None:
                f = R / f"fbp_{s}" / "first4.npy"
                arr = np.load(f)[0 if cname == "GT" else 1, img] if f.exists() else None
            else:
                f = R / pat.format(s=s) / "first4.npy"
                arr = np.load(f)[2, img] if f.exists() else None
            if arr is None:
                continue
            arr = arr.astype(np.float32)
            m = disk_mask(arr.shape[-1])
            if cname == "GT":
                ref = arr
            ax.imshow(np.where(m, arr, -1.0), cmap="gray", vmin=WIN[0], vmax=WIN[1])
            if ref is not None and cname != "GT":
                mse = np.mean((arr - ref)[m] ** 2)  # 与表格一致：只在重建圆内计算
                ax.text(4, 250, f"{10 * np.log10(4 / max(mse, 1e-12)):.1f} dB", color="yellow", fontsize=7)
            if i == 0:
                ax.set_title(cname, fontsize=9)
        axes[i, 0].text(-20, 128, sname.replace("$", "").replace(r"I_0{=}10^5", "I0=1e5"), rotation=90,
                        va="center", ha="right", fontsize=9)
    plt.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


def fig_nfe(R: Path, out: Path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4, 3))
    for m, name in GEN:
        xs, ys = [], []
        for nfe, d in ((20, f"nfe20_sv60_{m}"), (50, f"main_sv60_{m}"), (100, f"nfe100_sv60_{m}")):
            e = load_exp(R / d)
            if e:
                xs.append(nfe); ys.append(e[0]["metrics"]["psnr"]["mean"])
        if xs:
            ax.plot(xs, ys, "o-", label=name.replace("$", "").replace(r"\Pi", "Π"))
    ax.set_xlabel("NFE"); ax.set_ylabel("PSNR (dB)"); ax.set_xscale("log"); ax.set_xticks([20, 50, 100])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())  # 对数轴默认的次刻度标签会挤成 "3×10¹"
    ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax.set_title("60 views, test200", fontsize=9)
    plt.tight_layout(); fig.savefig(out); plt.close(fig)


def fig_tuning(R: Path, out: Path):
    """val 上引导强度 vs PSNR / 数据一致性（sv60）。"""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sel = json.load(open(R / "selected_params.json")) if (R / "selected_params.json").exists() else None
    if not sel:
        return
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    for ax, (m, name) in zip(axes, GEN):
        grid = sel["sv60"][m]["grid"]
        ax2 = ax.twinx()
        # dps 每个 t_start 一条曲线；pigdm / dc 只有 t_start=0
        for k, ts in enumerate(sorted({g["t_start"] for g in grid})):
            pts = sorted((g["strength"], g["val_psnr"]) for g in grid if g["t_start"] == ts)
            dcs = []
            for v, _ in pts:
                d = R / (f"tune_sv60_{m}_{v:g}" + (f"_ts{ts:g}" if ts else ""))
                dcs.append(load_exp(d)[0]["metrics"]["dc"]["mean"])
            lab = f"$t_s$={ts:g}" if m == "dps" else None
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=f"C{k}", label=lab)
            ax2.plot([p[0] for p in pts], dcs, "s--", color=f"C{k}", alpha=.5)
        ax.set_xscale("log"); ax.set_ylabel("val PSNR (dB), solid"); ax2.set_ylabel(r"$\|Ax-y\|/\|y\|$, dashed")
        ax.set_xlabel(r"$\tilde\lambda$" if m == "dc" else r"$\zeta$")
        ax.set_title(name.replace("$", "").replace(r"\Pi", "Π"), fontsize=9)
        if m == "dps":
            ax.legend(fontsize=7)
    plt.tight_layout(); fig.savefig(out); plt.close(fig)


def fig_uncertainty(R: Path, out: Path, img: int = 3):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    exps = [(name, R / f"unc8_sv60_{m}" / "first4.npy") for m, name in GEN]
    exps = [(n, f) for n, f in exps if f.exists()]
    if not exps:
        return
    fig, axes = plt.subplots(2, len(exps), figsize=(2.6 * len(exps), 5.2))
    axes = np.atleast_2d(axes).reshape(2, len(exps))
    for j, (name, f) in enumerate(exps):
        a = np.load(f).astype(np.float32)
        m = disk_mask(a.shape[-1])  # 圆外不受测量约束，误差与方差都不解释，置零显示
        err = np.where(m, np.abs(a[2, img] - a[0, img]), 0); std = np.where(m, a[3, img], 0)
        axes[0, j].imshow(err, cmap="magma", vmin=0, vmax=0.15); axes[0, j].set_title(name.replace("$", "").replace(r"\Pi", "Π") + " |error|", fontsize=8)
        axes[1, j].imshow(std, cmap="magma", vmin=0, vmax=0.15); axes[1, j].set_title("posterior std (K=8)", fontsize=8)
        for ax in axes[:, j]:
            ax.axis("off")
    plt.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results", type=Path)
    ap.add_argument("--index", default="results/index_test200.csv", type=Path)
    ap.add_argument("--paper", default="../papers/arxiv_phase1", type=Path)
    args = ap.parse_args()
    R = args.results
    (args.paper / "tables").mkdir(parents=True, exist_ok=True); (args.paper / "figures").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    groups = patient_groups(args.index, 200)

    summary = main_table(R, groups, rng, args.paper / "tables" / "main.tex")
    report = {"main_psnr": {f"{s}|{n}": v for (s, n), v in summary.items()}, "paired": {}}
    for s, _ in SETTINGS:
        for m, _ in GEN:
            for other in (f"fbp_{s}", f"fbpconvnet_{s}"):
                r = paired(R, f"main_{s}_{m}", other, groups, rng)
                if r:
                    report["paired"][f"{s}|{m}-{other.split('_')[0]}"] = {"diff": r[0], "ci": r[1]}
        for a, b in (("dc", "pigdm"), ("dc", "dps"), ("pigdm", "dps")):
            r = paired(R, f"main_{s}_{a}", f"main_{s}_{b}", groups, rng)
            if r:
                report["paired"][f"{s}|{a}-{b}"] = {"diff": r[0], "ci": r[1]}
    # 后验均值（K 次采样平均）的配对比较
    for s, k in (("sv60", 8), ("sv30", 4)):
        for a, b, cb in ((f"unc{k}_{s}_dc", f"unc{k}_{s}_pigdm", "psnr_mean_of_samples"),
                         (f"unc{k}_{s}_dc", f"fbpconvnet_{s}", "psnr"),
                         (f"unc{k}_{s}_pigdm", f"fbpconvnet_{s}", "psnr")):
            r = paired(R, a, b, groups, rng, col_a="psnr_mean_of_samples", col_b=cb)
            if r:
                report["paired"][f"{s}|mean{k}:{a.split('_')[-1]}-{b.split('_')[0] if 'fbpconvnet' in b else 'mean' + b.split('_')[-1]}"] = {
                    "diff": r[0], "ci": r[1]}
    json.dump(report, open(R / "analysis.json", "w"), indent=2)
    fig_qualitative(R, args.paper / "figures" / "qualitative.png")
    fig_nfe(R, args.paper / "figures" / "nfe.pdf")
    fig_tuning(R, args.paper / "figures" / "tuning.pdf")
    fig_uncertainty(R, args.paper / "figures" / "uncertainty.png")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
