"""Track 2-1（2D 测量条件化流微调 · 15 CU 硬止损门）实验编排器。

阶段（--stage，可按序传入多个）：
    profile   在 GPU 上运行 15 秒逐模块性能剖析
    train     训练 10k 步 ControlFlowUNet（支持断点续训与 Drive 同步）
    tune      在 val(50) 上为 la120 与 la90 选择最优推理方式（none / dc lam / cond_scale）
    gate      在 test200 上运行 la120 / la90 硬止损门控评测（外加 sv60 / ld1e5 跨协议泛化检查），
              输出 gate_report.json 与判定结论（通过阈值：la120 >= M3+2.0 dB 且 la90 >= M3+2.5 dB）

用法（在 Colab 仓库 code/ 目录下）：
    python scripts/run_track2_1.py --prior checkpoints/prior_256/ema_final.pt \
        --data-dir /content/data/processed --sync-dir /content/drive/MyDrive/flow-ct-guidance/track2_1 \
        --stage train tune gate
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

SETTINGS = {
    "la120": "configs/eval_la120.yaml",
    "la90": "configs/eval_la90.yaml",
    "sv60": "configs/eval_sparse60.yaml",
    "sv30": "configs/eval_sparse30.yaml",
    "ld1e5": "configs/eval_lowdose1e5.yaml",
}
# 路线图 v2 锁定的硬止损基线（test200）
M3_BASE = {"la120": 29.32, "la90": 24.38}
GATE_DELTA = {"la120": 2.0, "la90": 2.5}
FBPCONV_BASE = {"la120": 32.44, "la90": 29.98}

N_TUNE = 50
RESULTS = Path("results_t2_1")


def sync_dir_tree(src: Path, dst: Path | None):
    if dst is None or not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        if p.is_file() and p.suffix in (".json", ".csv", ".yaml", ".png", ".npy"):
            shutil.copy2(p, dst / p.name)


def run_eval(cmd: list[str], out: Path, sync_root: Path | None):
    if (out / "metrics.json").exists():
        print(f"[SKIP] {out}", flush=True)
        return
    if out.exists():
        shutil.rmtree(out)
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run([sys.executable, "-u", *cmd], check=True)
    sync_dir_tree(out, sync_root / "results" / out.name if sync_root else None)


def sample_cmd(
    prior: str,
    control_ckpt: str,
    setting: str,
    guidance: str,
    out: Path,
    data_dir: str,
    batch_size: int = 16,
    nfe: int = 50,
    n_samples: int = 1,
    split: str | None = None,
    n_test: int | None = None,
    dc_lam: float | None = None,
    dc_iters: int = 5,
    cond_scale: float = 1.0,
    cfg_scale: float = 1.0,
    noise_model: str = "none",
    noise_scale: float = 1.0,
    dc_schedule: str = "const",
    no_lpips: bool = False,
) -> list[str]:
    cmd = [
        "src/sample_posterior.py",
        "--ckpt", prior,
        "--control-ckpt", control_ckpt,
        "--cond-scale", str(cond_scale),
        "--cfg-scale", str(cfg_scale),
        "--config", SETTINGS[setting],
        "--guidance", guidance,
        "--nfe", str(nfe),
        "--n-samples", str(n_samples),
        "--dc-iters", str(dc_iters),
        "--noise-model", noise_model,
        "--noise-scale", str(noise_scale),
        "--dc-schedule", dc_schedule,
        "--data-dir", data_dir,
        "--batch-size", str(batch_size),
        "--out", str(out),
    ]
    if split:
        cmd += ["--split", split]
    if n_test:
        cmd += ["--n-test", str(n_test)]
    if dc_lam is not None:
        cmd += ["--dc-lam", str(dc_lam)]
    if no_lpips:
        cmd += ["--no-lpips"]
    return cmd


def metrics_of(out: Path) -> dict:
    d = json.loads((out / "metrics.json").read_text())
    m = d["metrics"]
    return {
        "psnr": float(m["psnr"]["mean"]),
        "ssim": float(m["ssim"]["mean"]),
        "lpips": float(m["lpips"]["mean"]) if "lpips" in m else None,
        "dc": float(m["dc"]["mean"]),
        "minutes": float(d.get("minutes", 0.0)),
    }


def load_summary() -> dict:
    p = RESULTS / "gate_report.json"
    return json.loads(p.read_text()) if p.exists() else {}


def save_summary(data: dict, sync_root: Path | None):
    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / "gate_report.json"
    p.write_text(json.dumps(data, indent=2))
    if sync_root:
        (sync_root / "results").mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, sync_root / "results" / p.name)


def stage_profile(args):
    cmd = [
        "src/train_control_flow.py",
        "--config", args.train_config,
        "--prior", args.prior,
        "--data-dir", args.data_dir,
        "--profile",
    ]
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run([sys.executable, "-u", *cmd], check=True)


def stage_train(args, sync_root: Path | None):
    ckpt_dir = Path(args.ckpt_dir)
    final_ckpt = ckpt_dir / "control_final.pt"
    sync_ckpt = sync_root / "ckpt" if sync_root else None
    if final_ckpt.exists():
        print(f"[SKIP] 已存在最终权重 {final_ckpt}", flush=True)
        return
    if sync_ckpt and (sync_ckpt / "control_final.pt").exists():
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sync_ckpt / "control_final.pt", final_ckpt)
        print(f"[RESTORE] 从同步目录恢复最终权重 {final_ckpt}", flush=True)
        return
    cmd = [
        "src/train_control_flow.py",
        "--config", args.train_config,
        "--prior", args.prior,
        "--data-dir", args.data_dir,
        "--out-dir", str(ckpt_dir),
        "--resume",
    ]
    if args.max_steps:
        cmd += ["--max-steps", str(args.max_steps)]
    if sync_ckpt:
        cmd += ["--sync-dir", str(sync_ckpt)]
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run([sys.executable, "-u", *cmd], check=True)


def stage_tune(args, sync_root: Path | None):
    """严格在 val(50) 上选择 la120 / la90 的推理参数（不触碰 test200）。"""
    ctrl = str(Path(args.ckpt_dir) / "control_final.pt")
    rep = load_summary()
    tune = rep.setdefault("tune_val50", {})

    for setting in ("la120", "la90"):
        grid = {}
        # 1) 纯条件 ODE（NFE=50 与 NFE=25）
        for nfe in (25, 50):
            key = f"none_nfe{nfe}"
            out = RESULTS / f"tune_{setting}_cond_{key}"
            run_eval(
                sample_cmd(args.prior, ctrl, setting, "none", out, args.data_dir, args.eval_bs,
                           nfe=nfe, split="val", n_test=N_TUNE, no_lpips=True),
                out, sync_root,
            )
            grid[key] = {"guidance": "none", "nfe": nfe, "dc_lam": None, **metrics_of(out)}

        # 2) 条件流 + Prox-DC（扫描 dc_lam ∈ {0.1, 0.3, 1.0, 3.0}）
        for lam in (0.1, 0.3, 1.0, 3.0):
            key = f"dc_l{lam:g}"
            out = RESULTS / f"tune_{setting}_cond_{key}"
            run_eval(
                sample_cmd(args.prior, ctrl, setting, "dc", out, args.data_dir, args.eval_bs,
                           nfe=50, split="val", n_test=N_TUNE, dc_lam=lam, no_lpips=True),
                out, sync_root,
            )
            grid[key] = {"guidance": "dc", "nfe": 50, "dc_lam": lam, **metrics_of(out)}

        best_dc = max((v for k, v in grid.items() if k.startswith("dc_")), key=lambda d: d["psnr"])
        best_all = max(grid.values(), key=lambda d: d["psnr"])
        tune[setting] = {"best_dc": best_dc, "best_all": best_all, "grid": grid}
        save_summary(rep, sync_root)

    print("\n=== T2-1 VAL(50) TUNE SUMMARY ===", json.dumps(tune, indent=2), flush=True)


def stage_gate(args, sync_root: Path | None):
    """在 test200 上运行 15 CU 硬止损门控评测（la120 / la90 + sv60 / ld1e5 跨协议泛化）。"""
    ctrl = str(Path(args.ckpt_dir) / "control_final.pt")
    rep = load_summary()
    tune = rep.get("tune_val50", {})
    test_res = rep.setdefault("test200", {})

    # 1) 核心止损门：la120 与 la90（同时记录纯条件 ODE 与最优条件+Prox-DC）
    verdicts = {}
    for setting in ("la120", "la90"):
        best_lam = tune.get(setting, {}).get("best_dc", {}).get("dc_lam", 0.3)

        name_none = f"main_{setting}_cond_none"
        out_none = RESULTS / name_none
        run_eval(
            sample_cmd(args.prior, ctrl, setting, "none", out_none, args.data_dir, args.eval_bs, nfe=50),
            out_none, sync_root,
        )
        test_res[name_none] = metrics_of(out_none)

        name_dc = f"main_{setting}_cond_dc"
        out_dc = RESULTS / name_dc
        run_eval(
            sample_cmd(args.prior, ctrl, setting, "dc", out_dc, args.data_dir, args.eval_bs,
                       nfe=50, dc_lam=best_lam),
            out_dc, sync_root,
        )
        test_res[name_dc] = {"dc_lam": best_lam, **metrics_of(out_dc)}

        best_psnr = max(test_res[name_none]["psnr"], test_res[name_dc]["psnr"])
        gain_vs_m3 = best_psnr - M3_BASE[setting]
        gain_vs_fbpconv = best_psnr - FBPCONV_BASE[setting]
        passed = gain_vs_m3 >= GATE_DELTA[setting]
        verdicts[setting] = {
            "best_psnr": round(best_psnr, 3),
            "m3_baseline": M3_BASE[setting],
            "fbpconvnet_baseline": FBPCONV_BASE[setting],
            "gain_vs_m3": round(gain_vs_m3, 3),
            "gain_vs_fbpconvnet": round(gain_vs_fbpconv, 3),
            "required_gain": GATE_DELTA[setting],
            "passed": passed,
        }
        save_summary(rep, sync_root)

    # 2) 轻量跨协议泛化检验：sv60（cond + dc）与 ld1e5（cond + Poisson PWLS-DC）
    for name, cmd in [
        ("main_sv60_cond_dc", sample_cmd(args.prior, ctrl, "sv60", "dc", RESULTS / "main_sv60_cond_dc",
                                         args.data_dir, args.eval_bs, nfe=50, dc_lam=0.3)),
        ("main_ld1e5_cond_dc_poisson", sample_cmd(args.prior, ctrl, "ld1e5", "dc",
                                                  RESULTS / "main_ld1e5_cond_dc_poisson",
                                                  args.data_dir, args.eval_bs, nfe=50, dc_lam=0.3,
                                                  noise_model="poisson", dc_schedule="const")),
    ]:
        out = RESULTS / name
        run_eval(cmd, out, sync_root)
        test_res[name] = metrics_of(out)

    rep["gate_verdict"] = {
        "overall_passed": all(v["passed"] for v in verdicts.values()),
        "details": verdicts,
    }
    save_summary(rep, sync_root)
    print("\n=== T2-1 15 CU HARD STOP-LOSS GATE VERDICT ===", flush=True)
    print(json.dumps(rep["gate_verdict"], indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--train-config", default="configs/control_flow_t2_1.yaml")
    ap.add_argument("--ckpt-dir", default="checkpoints/control_flow_t2_1")
    ap.add_argument("--sync-dir", default=None)
    ap.add_argument("--eval-bs", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--stage", nargs="+", required=True, choices=["profile", "train", "tune", "gate"])
    args = ap.parse_args()
    sync_root = Path(args.sync_dir) if args.sync_dir else None

    for st in args.stage:
        print(f"\n[STAGE] {st}", flush=True)
        if st == "profile":
            stage_profile(args)
        elif st == "train":
            stage_train(args, sync_root)
        elif st == "tune":
            stage_tune(args, sync_root)
        elif st == "gate":
            stage_gate(args, sync_root)


if __name__ == "__main__":
    main()
