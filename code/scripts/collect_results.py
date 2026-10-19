"""汇总 results/*/metrics.json 为一张表（CSV + Markdown），供周报与论文主表使用。

用法：python scripts/collect_results.py [--root results] [--prefix exp_]
输出：<root>/summary.csv, <root>/summary.md；只读 metrics.json，不改动任何实验目录。
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def setting_name(phys: dict) -> str:
    """把 physics 配置映射成可读的设置名，如 sv60 / ld1e4 / la120-sv60。"""
    n = phys.get("n_angles", 180)
    rng = phys.get("angle_range")
    tag = f"sv{n}" if phys.get("i0") is None else f"ld{phys['i0']:.0e}".replace("+0", "")
    if rng is not None and abs(rng - 3.141592653589793) > 1e-6:
        tag = f"la{round(rng * 180 / 3.141592653589793)}-{tag}"
    if phys.get("angle_offset"):
        tag += f"-off{phys['angle_offset']:.4f}"
    return tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results", type=Path)
    ap.add_argument("--prefix", default="", help="只汇总名字以此开头的实验目录")
    args = ap.parse_args()

    rows = []
    for mj in sorted(args.root.glob(f"{args.prefix}*/metrics.json")):
        r = json.load(open(mj))
        m = r["metrics"]

        def g(k, f="mean"):
            return m[k][f] if k in m else None

        rows.append({
            "exp": mj.parent.name,
            "method": r.get("guidance") or r.get("method"),
            "setting": setting_name(r.get("physics", {})),
            # dc 的强度参数是相对 λ（dc_lam），其余方法是 ζ；表中同列显示
            "nfe": r.get("nfe"), "zeta": r.get("dc_lam") if r.get("guidance") == "dc" else r.get("zeta"),
            "n_samples": r.get("n_samples"),
            "n_test": g("psnr", "n"),
            "psnr": g("psnr"), "psnr_std": g("psnr", "std"), "ssim": g("ssim"),
            "lpips": g("lpips"), "dc": g("dc"),
            "psnr_mean_of_samples": g("psnr_mean_of_samples"), "pixel_std": g("pixel_std"),
            "minutes": r.get("minutes"),
        })
    if not rows:
        print("没有找到 metrics.json")
        return
    keys = list(rows[0])
    with open(args.root / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)

    def fmt(v, nd=3):
        return "" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))

    lines = ["| exp | method | setting | NFE | ζ/λ | K | PSNR | SSIM | LPIPS | DC | min |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['exp']} | {r['method']} | {r['setting']} | {fmt(r['nfe'])} | {fmt(r['zeta'], 2)} | "
                     f"{fmt(r['n_samples'])} | {fmt(r['psnr'], 2)} ± {fmt(r['psnr_std'], 2)} | {fmt(r['ssim'])} | "
                     f"{fmt(r['lpips'])} | {fmt(r['dc'], 4)} | {fmt(r['minutes'], 1)} |")
    (args.root / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
