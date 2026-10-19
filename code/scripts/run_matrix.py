"""Phase 1 对比矩阵编排：按阶段顺序调用 fbp_eval / sample_posterior，已有 metrics.json 的实验自动跳过（可断点续跑）。

阶段（--stage，可多次指定，按给出顺序执行）：
    fbp      B0 FBP，5 个设置 × test200
    tune     在 val（等间隔 50 张）上网格搜索引导强度：sv60 全网格，ld1e5 缩小网格；结果写 results/tune_*
    select   读取 tune 结果，按 val PSNR 选每个方法的 ζ/λ，写 results/selected_params.json
    main     主表：4 设置 × 3 方法 × NFE 50 × 单样本 × test200（sv30/sv90 沿用 sv60 选出的参数）
    nfe      NFE 消融：sv60 × 3 方法 × NFE {20, 100}
    unc      不确定性：sv60 × 3 方法 × 8 次采样；sv30 × 3 方法 × 4 次采样
    ld1e4    极端低剂量附加：I0=1e4 × 3 方法（沿用 ld1e5 参数）
    fbpconvnet  B1 监督基线：sv30/sv60/sv90/ld1e5 各训 10k 步；权重留在 checkpoints/，小文件复制到 results/fbpconvnet_<设置>

用法（远端，01_diffusion_inverse 目录下）：
    python scripts/run_matrix.py --ckpt checkpoints/prior_256/ema_final.pt --stage fbp tune select main
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

SETTINGS = {"sv30": "configs/eval_sparse30.yaml", "sv60": "configs/eval_sparse60.yaml",
            "sv90": "configs/eval_sparse90.yaml", "ld1e5": "configs/eval_lowdose1e5.yaml",
            "ld1e4": "configs/eval_lowdose.yaml"}
METHODS = ("dps", "pigdm", "dc")
# 网格元素为 (强度, t_start)：强度对 dps/pigdm 是 ζ，对 dc 是相对 λ（单位 tr(AᵀA)/N²）。
# dps 的梯度在 t≈0 处比后续步大约 60 倍（纯噪声处网络 Jacobian 大），从 t=0 引导时大 ζ 首步即发散、
# 小 ζ 后续步又几乎无效（预检：t_start=0 时最好 13.8 dB，t_start=0.2 时 20.3 dB，val 16 张 / NFE 20），
# 因此对 dps 额外搜索引导起始时刻；pigdm / dc 的修正项自限制，从 t=0 开始。
# 第一轮网格（2026-10-19 16:10）有 5/6 个最优值落在边界，第二轮按以下网格外扩（已跑过的点自动复用）。
GRID = {"sv60": {"dps": [(z, ts) for z in (3, 10, 30, 100, 300) for ts in (0.1, 0.2, 0.3, 0.4, 0.5)],
                 "pigdm": [(z, 0.0) for z in (0.5, 1, 2)],
                 "dc": [(z, 0.0) for z in (0.01, 0.03, 0.1, 0.3, 1, 3)]},
        # 第二轮后 ld1e5 的 dps(t_s=0.4) 与 pigdm(ζ=0.1) 仍在边界，第三轮再外扩
        "ld1e5": {"dps": [(z, ts) for z in (10, 30, 100) for ts in (0.2, 0.3, 0.4, 0.5)],
                  "pigdm": [(z, 0.0) for z in (0.03, 0.05, 0.1, 0.25, 0.5, 1)],
                  "dc": [(z, 0.0) for z in (1, 3, 10, 30, 100)]}}
N_TUNE = 50
RESULTS = Path("results")
SELECTED = RESULTS / "selected_params.json"


def run(cmd: list[str], expect: dict | None = None):
    """执行一个实验；out 目录已有 metrics.json 且其中参数与 expect 一致时视为完成并跳过。

    参数不一致（例如重新选参后强度变了）时，把旧目录改名为 <out>_stale<n> 留档后重跑。失败时直接抛出，由调用者终止整轮。
    """
    out = Path(cmd[cmd.index("--out") + 1])
    mj = out / "metrics.json"
    if mj.exists():
        old = json.load(open(mj))
        diff = {k: (old.get(k), v) for k, v in (expect or {}).items() if old.get(k) != v}
        if not diff:
            print(f"[SKIP] {out}", flush=True)
            return
        n = 1
        while out.with_name(f"{out.name}_stale{n}").exists():
            n += 1
        out.rename(out.with_name(f"{out.name}_stale{n}"))
        print(f"[STALE] {out} 参数变化 {diff}，旧结果改名为 _stale{n}", flush=True)
    elif out.exists():
        shutil.rmtree(out)  # 中断留下的不完整目录（没有 metrics.json）
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run([sys.executable, "-u", *cmd], check=True)


def strength_arg(method: str, value: float) -> list[str]:
    return ["--dc-lam", str(value)] if method == "dc" else ["--zeta", str(value)]


def sample(ckpt, setting, method, param, out, nfe=50, n_samples=1, split=None, n_test=None):
    """param = (强度, t_start)。"""
    value, t_start = param
    cmd = ["src/sample_posterior.py", "--ckpt", ckpt, "--config", SETTINGS[setting], "--guidance", method,
           "--nfe", str(nfe), "--n-samples", str(n_samples), "--out", str(out), *strength_arg(method, value),
           "--t-start", str(t_start)]
    if split:
        cmd += ["--split", split]
    if n_test:
        cmd += ["--n-test", str(n_test)]
    strength_key = "dc_lam" if method == "dc" else "zeta"
    run(cmd, expect={strength_key: float(value), "t_start_guidance": float(t_start), "nfe": nfe,
                     "n_samples": n_samples})


def tune_dir(setting, method, param):
    value, t_start = param
    return RESULTS / (f"tune_{setting}_{method}_{value:g}" + (f"_ts{t_start:g}" if t_start else ""))


def selected() -> dict:
    if not SELECTED.exists():
        raise SystemExit("缺少 results/selected_params.json，请先运行 --stage tune select")
    return json.load(open(SELECTED))


def param_for(sel: dict, setting: str, method: str) -> tuple[float, float]:
    """sv30/sv90 沿用 sv60 的参数，ld1e4 沿用 ld1e5 的参数（预算内不对每个设置单独调参，论文中注明）。"""
    src = {"sv30": "sv60", "sv90": "sv60", "ld1e4": "ld1e5"}.get(setting, setting)
    return tuple(sel[src][method]["param"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--stage", nargs="+", required=True,
                    choices=["fbp", "tune", "select", "main", "nfe", "unc", "ld1e4", "fbpconvnet"])
    args = ap.parse_args()
    ck = args.ckpt

    for stage in args.stage:
        print(f"[START] stage={stage}", flush=True)
        if stage == "fbp":
            for s, cfg in SETTINGS.items():
                run(["src/baselines/fbp_eval.py", "--config", cfg, "--out", str(RESULTS / f"fbp_{s}")])
        elif stage == "tune":
            for s, grid in GRID.items():
                for m, values in grid.items():
                    for prm in values:
                        sample(ck, s, m, prm, tune_dir(s, m, prm), split="val", n_test=N_TUNE)
        elif stage == "select":
            sel = {}
            for s, grid in GRID.items():
                sel[s] = {}
                for m, values in grid.items():
                    scores = {prm: json.load(open(tune_dir(s, m, prm) / "metrics.json"))["metrics"]["psnr"]["mean"]
                              for prm in values}
                    best = max(scores, key=scores.get)
                    # 强度最优值落在网格边界时提示，说明网格可能需要外扩
                    strengths = sorted({p[0] for p in values}); starts = sorted({p[1] for p in values})
                    edge = best[0] in (strengths[0], strengths[-1]) or (
                        len(starts) > 1 and best[1] in (starts[0], starts[-1]))
                    sel[s][m] = {"param": list(best), "val_psnr": scores[best], "at_edge": edge,
                                 "grid": [{"strength": p[0], "t_start": p[1], "val_psnr": v} for p, v in scores.items()]}
                    print(f"[METRIC] {s} {m}: best={best} val_psnr={scores[best]:.2f}{'  [EDGE]' if edge else ''}")
            json.dump(sel, open(SELECTED, "w"), indent=2)
        elif stage == "main":
            sel = selected()
            for s in ("sv30", "sv60", "sv90", "ld1e5"):
                for m in METHODS:
                    sample(ck, s, m, param_for(sel, s, m), RESULTS / f"main_{s}_{m}")
        elif stage == "nfe":
            sel = selected()
            for nfe in (20, 100):
                for m in METHODS:
                    sample(ck, "sv60", m, param_for(sel, "sv60", m), RESULTS / f"nfe{nfe}_sv60_{m}", nfe=nfe)
        elif stage == "unc":
            sel = selected()
            for s, k in (("sv60", 8), ("sv30", 4)):
                for m in METHODS:
                    sample(ck, s, m, param_for(sel, s, m), RESULTS / f"unc{k}_{s}_{m}", n_samples=k)
        elif stage == "ld1e4":
            sel = selected()
            for m in METHODS:
                sample(ck, "ld1e4", m, param_for(sel, "ld1e4", m), RESULTS / f"main_ld1e4_{m}")
        elif stage == "fbpconvnet":
            for s in ("sv30", "sv60", "sv90", "ld1e5"):
                dst = RESULTS / f"fbpconvnet_{s}"
                if (dst / "metrics.json").exists():
                    print(f"[SKIP] {dst}", flush=True)
                    continue
                cfg = f"configs/fbpconvnet_{s}.yaml"
                print("[RUN] train_fbpconvnet", cfg, flush=True)
                subprocess.run([sys.executable, "-u", "src/baselines/train_fbpconvnet.py", "--config", cfg], check=True)
                dst.mkdir(parents=True, exist_ok=True)
                src_dir = Path("checkpoints") / f"fbpconvnet_{s}"
                for name in ("metrics.json", "per_image.csv", "samples.png", "first4.npy"):
                    shutil.copy2(src_dir / name, dst / name)
                shutil.copy2(cfg, dst / "config.yaml")
        print(f"[DONE] stage={stage}", flush=True)


if __name__ == "__main__":
    main()
