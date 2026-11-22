"""Track 1（Paper 1 短板补齐）实验编排器：支持断点续跑、自动 Drive 同步与结果汇总。

阶段（--stage，可按序传入多个）：
    t1_1          Poisson 噪声感知流求解器（M2 scalar / M2 poisson / M3 pwls-const / M3 pwls-rt2）
                  在 ld1e5 val(50) 选参，在 ld1e5 与 ld1e4 test200 上评测（含零样本迁移）
    t1_2          早 M3 晚 M2 混合调度（sv60 t_sw in {0.2, 0.4, 0.6} + 稀疏全设置主表）
                  + 限角 la120 M2 下界复核（zeta in {0.05, 0.1, 0.25}）
    t1_3_grad     用最终 120k 权重复测 M1 梯度曲线与 t_start=0 上限
    t1_3_fbp_la   补训限角监督基线 fbpconvnet_la120 / fbpconvnet_la90（各 10k 步）
    t1_3_fbp_30k  对 sv60 与 ld1e5 跑 30k 步长训 FBPConvNet（移除 3.7 epoch 下界限制）

用法（在 Colab 仓库 code/ 目录下）：
    python scripts/run_track1.py --prior checkpoints/prior_256/ema_final.pt \
        --data-dir /content/data/processed --sync-dir /content/drive/MyDrive/flow-ct-guidance/track1 \
        --stage t1_1 t1_2 t1_3_grad t1_3_fbp_la t1_3_fbp_30k
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

SETTINGS = {
    "sv30": "configs/eval_sparse30.yaml",
    "sv60": "configs/eval_sparse60.yaml",
    "sv90": "configs/eval_sparse90.yaml",
    "la120": "configs/eval_la120.yaml",
    "la90": "configs/eval_la90.yaml",
    "ld1e5": "configs/eval_lowdose1e5.yaml",
    "ld1e4": "configs/eval_lowdose.yaml",
}
N_TUNE = 50
RESULTS = Path("results_t1")


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


def sample_cmd(prior: str, setting: str, guidance: str, out: Path, data_dir: str | None,
               batch_size: int | None = None, nfe: int = 50, n_samples: int = 1,
               split: str | None = None, n_test: int | None = None,
               zeta: float | None = None, dc_lam: float | None = None,
               dc_iters: int | None = None, pigdm_cg_iters: int | None = None,
               t_switch: float | None = None, noise_model: str = "none",
               noise_scale: float = 1.0, dc_schedule: str = "const") -> list[str]:
    cmd = ["src/sample_posterior.py", "--ckpt", prior, "--config", SETTINGS[setting],
           "--guidance", guidance, "--nfe", str(nfe), "--n-samples", str(n_samples),
           "--noise-model", noise_model, "--noise-scale", str(noise_scale),
           "--dc-schedule", dc_schedule, "--out", str(out)]
    if data_dir:
        cmd += ["--data-dir", data_dir]
    if batch_size:
        cmd += ["--batch-size", str(batch_size)]
    if split:
        cmd += ["--split", split]
    if n_test:
        cmd += ["--n-test", str(n_test)]
    if zeta is not None:
        cmd += ["--zeta", str(zeta)]
    if dc_lam is not None:
        cmd += ["--dc-lam", str(dc_lam)]
    if dc_iters is not None:
        cmd += ["--dc-iters", str(dc_iters)]
    if pigdm_cg_iters is not None:
        cmd += ["--pigdm-cg-iters", str(pigdm_cg_iters)]
    if t_switch is not None:
        cmd += ["--t-switch", str(t_switch)]
    return cmd


def psnr_of(out: Path) -> float:
    return float(json.loads((out / "metrics.json").read_text())["metrics"]["psnr"]["mean"])


def minutes_of(out: Path) -> float:
    return float(json.loads((out / "metrics.json").read_text()).get("minutes", 0.0))


def load_summary() -> dict:
    p = RESULTS / "selected_t1.json"
    return json.loads(p.read_text()) if p.exists() else {}


def save_summary(data: dict, sync_root: Path | None):
    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / "selected_t1.json"
    p.write_text(json.dumps(data, indent=2))
    if sync_root:
        (sync_root / "results").mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, sync_root / "results" / p.name)


def stage_t1_1(args, sync_root: Path | None):
    """T1-1: Poisson 噪声感知流求解器（M2 scalar / M2 poisson / M3 pwls-const / M3 pwls-rt2）。"""
    sel = load_summary()
    t11 = sel.setdefault("t1_1", {})

    # 1) M2 scalar noise model（基于前期探针已排除 z=1.0 次优区间，聚焦 z∈{0.25,0.5} × ns∈{1,2,4}）
    m2_scalar_scores = {}
    for z in (0.25, 0.5):
        for ns in (1.0, 2.0, 4.0):
            key = f"z{z:g}_ns{ns:g}"
            out = RESULTS / f"tune_ld1e5_pigdm_scalar_{key}"
            run_eval(sample_cmd(args.prior, "ld1e5", "pigdm", out, args.data_dir, args.eval_bs,
                                split="val", n_test=N_TUNE, zeta=z, pigdm_cg_iters=5,
                                noise_model="scalar", noise_scale=ns),
                     out, sync_root)
            m2_scalar_scores[key] = {"zeta": z, "noise_scale": ns, "val_psnr": psnr_of(out)}
    best_m2s = max(m2_scalar_scores.values(), key=lambda d: d["val_psnr"])
    t11["m2_scalar"] = {"best": best_m2s, "grid": m2_scalar_scores}
    save_summary(sel, sync_root)

    # 2) M2 per-ray Poisson noise model + Jacobi PCG（与 Phase 1 保持相同 5 步 CG）
    m2_pois_scores = {}
    for z in (0.25, 0.5):
        for ns in (0.5, 1.0, 2.0, 4.0):
            key = f"z{z:g}_ns{ns:g}"
            out = RESULTS / f"tune_ld1e5_pigdm_poisson_{key}"
            run_eval(sample_cmd(args.prior, "ld1e5", "pigdm", out, args.data_dir, args.eval_bs,
                                split="val", n_test=N_TUNE, zeta=z, pigdm_cg_iters=5,
                                noise_model="poisson", noise_scale=ns),
                     out, sync_root)
            m2_pois_scores[key] = {"zeta": z, "noise_scale": ns, "val_psnr": psnr_of(out)}
    best_m2p = max(m2_pois_scores.values(), key=lambda d: d["val_psnr"])
    t11["m2_poisson"] = {"best": best_m2p, "grid": m2_pois_scores}
    save_summary(sel, sync_root)

    # 3) M3 PWLS weights + const schedule (relative=True; 空气射线高权使 w_mean 偏大，聚焦低 lam 区间)
    m3_const_scores = {}
    for lam in (0.03, 0.1, 0.3, 1.0):
        out = RESULTS / f"tune_ld1e5_dc_pwls_const_l{lam:g}"
        run_eval(sample_cmd(args.prior, "ld1e5", "dc", out, args.data_dir, args.eval_bs,
                            split="val", n_test=N_TUNE, dc_lam=lam, dc_iters=5,
                            noise_model="poisson", dc_schedule="const"),
                 out, sync_root)
        m3_const_scores[str(lam)] = {"dc_lam": lam, "val_psnr": psnr_of(out)}
    best_m3c = max(m3_const_scores.values(), key=lambda d: d["val_psnr"])
    t11["m3_pwls_const"] = {"best": best_m3c, "grid": m3_const_scores}
    save_summary(sel, sync_root)

    # 4) M3 exact Bayesian PWLS-MAP: W = Sigma_y^{-1}, lambda(t) = lam0 / r_t^2 (relative=False) + Jacobi PCG
    m3_rt2_scores = {}
    for lam in (30.0, 100.0, 300.0, 1000.0):
        out = RESULTS / f"tune_ld1e5_dc_pwls_rt2_l{lam:g}"
        run_eval(sample_cmd(args.prior, "ld1e5", "dc", out, args.data_dir, args.eval_bs,
                            split="val", n_test=N_TUNE, dc_lam=lam, dc_iters=5,
                            noise_model="poisson", dc_schedule="rt2"),
                 out, sync_root)
        m3_rt2_scores[str(lam)] = {"dc_lam": lam, "val_psnr": psnr_of(out)}
    best_m3r = max(m3_rt2_scores.values(), key=lambda d: d["val_psnr"])
    if best_m3r["dc_lam"] == 30.0:
        for lam in (3.0, 10.0):
            out = RESULTS / f"tune_ld1e5_dc_pwls_rt2_l{lam:g}"
            run_eval(sample_cmd(args.prior, "ld1e5", "dc", out, args.data_dir, args.eval_bs,
                                split="val", n_test=N_TUNE, dc_lam=lam, dc_iters=5,
                                noise_model="poisson", dc_schedule="rt2"),
                     out, sync_root)
            m3_rt2_scores[str(lam)] = {"dc_lam": lam, "val_psnr": psnr_of(out)}
    elif best_m3r["dc_lam"] == 1000.0:
        for lam in (3000.0, 10000.0):
            out = RESULTS / f"tune_ld1e5_dc_pwls_rt2_l{lam:g}"
            run_eval(sample_cmd(args.prior, "ld1e5", "dc", out, args.data_dir, args.eval_bs,
                                split="val", n_test=N_TUNE, dc_lam=lam, dc_iters=5,
                                noise_model="poisson", dc_schedule="rt2"),
                     out, sync_root)
            m3_rt2_scores[str(lam)] = {"dc_lam": lam, "val_psnr": psnr_of(out)}
    best_m3r = max(m3_rt2_scores.values(), key=lambda d: d["val_psnr"])
    t11["m3_pwls_rt2"] = {"best": best_m3r, "grid": m3_rt2_scores}
    save_summary(sel, sync_root)

    # 5) test200 主评：ld1e5 与 ld1e4（完全零样本迁移，不针对 ld1e4 重调）
    test_results = {}
    for setting in ("ld1e5", "ld1e4"):
        runs = [
            (f"main_{setting}_pigdm_scalar",
             sample_cmd(args.prior, setting, "pigdm", RESULTS / f"main_{setting}_pigdm_scalar",
                        args.data_dir, args.eval_bs, zeta=best_m2s["zeta"], pigdm_cg_iters=5,
                        noise_model="scalar", noise_scale=best_m2s["noise_scale"])),
            (f"main_{setting}_pigdm_poisson",
             sample_cmd(args.prior, setting, "pigdm", RESULTS / f"main_{setting}_pigdm_poisson",
                        args.data_dir, args.eval_bs, zeta=best_m2p["zeta"], pigdm_cg_iters=5,
                        noise_model="poisson", noise_scale=best_m2p["noise_scale"])),
            (f"main_{setting}_dc_pwls_const",
             sample_cmd(args.prior, setting, "dc", RESULTS / f"main_{setting}_dc_pwls_const",
                        args.data_dir, args.eval_bs, dc_lam=best_m3c["dc_lam"], dc_iters=5,
                        noise_model="poisson", dc_schedule="const")),
            (f"main_{setting}_dc_pwls_rt2",
             sample_cmd(args.prior, setting, "dc", RESULTS / f"main_{setting}_dc_pwls_rt2",
                        args.data_dir, args.eval_bs, dc_lam=best_m3r["dc_lam"], dc_iters=5,
                        noise_model="poisson", dc_schedule="rt2")),
        ]
        for name, cmd in runs:
            out = RESULTS / name
            run_eval(cmd, out, sync_root)
            test_results[name] = {"psnr": psnr_of(out), "minutes": minutes_of(out)}
    t11["test200"] = test_results
    save_summary(sel, sync_root)
    print("\n=== T1-1 SUMMARY ===", json.dumps(test_results, indent=2), flush=True)


def stage_t1_2(args, sync_root: Path | None):
    """T1-2: 早 M3 晚 M2 混合调度（sv60/sv30/sv90）+ 限角 la120 M2 下界复核。"""
    sel = load_summary()
    t12 = sel.setdefault("t1_2", {})

    # 1) sv60 val 扫 t_switch in {0.2, 0.4, 0.6}（使用 Phase 1 已选定参数 dc_lam=0.3, zeta=1.0）
    hyb_val = {}
    for tsw in (0.2, 0.4, 0.6):
        out = RESULTS / f"tune_sv60_hybrid_tsw{tsw:g}"
        run_eval(sample_cmd(args.prior, "sv60", "hybrid", out, args.data_dir, args.eval_bs,
                            split="val", n_test=N_TUNE, zeta=1.0, dc_lam=0.3, t_switch=tsw),
                 out, sync_root)
        hyb_val[str(tsw)] = {"t_switch": tsw, "val_psnr": psnr_of(out), "minutes": minutes_of(out)}
    best_tsw = max(hyb_val.values(), key=lambda d: d["val_psnr"])["t_switch"]
    t12["hybrid_sv60_val"] = hyb_val
    t12["best_t_switch"] = best_tsw
    save_summary(sel, sync_root)

    # 在 sv60 test200 上跑全部 3 个 t_switch 以便完整画表/对比速度，再把最佳 t_switch 迁移到 sv30/sv90
    hyb_test = {}
    for tsw in (0.2, 0.4, 0.6):
        name = f"main_sv60_hybrid_tsw{tsw:g}"
        out = RESULTS / name
        run_eval(sample_cmd(args.prior, "sv60", "hybrid", out, args.data_dir, args.eval_bs,
                            zeta=1.0, dc_lam=0.3, t_switch=tsw), out, sync_root)
        hyb_test[name] = {"psnr": psnr_of(out), "minutes": minutes_of(out)}
    # 同机复测纯 M2（sv60）耗时以提供同硬件公平加速比（结果与原 Phase 1 应一致）
    out_m2_bench = RESULTS / "main_sv60_pigdm_bench"
    run_eval(sample_cmd(args.prior, "sv60", "pigdm", out_m2_bench, args.data_dir, args.eval_bs, zeta=1.0),
             out_m2_bench, sync_root)
    hyb_test["main_sv60_pigdm_bench"] = {"psnr": psnr_of(out_m2_bench), "minutes": minutes_of(out_m2_bench)}

    for s in ("sv30", "sv90"):
        name = f"main_{s}_hybrid_tsw{best_tsw:g}"
        out = RESULTS / name
        run_eval(sample_cmd(args.prior, s, "hybrid", out, args.data_dir, args.eval_bs,
                            zeta=1.0, dc_lam=0.3, t_switch=best_tsw), out, sync_root)
        hyb_test[name] = {"psnr": psnr_of(out), "minutes": minutes_of(out)}
    t12["hybrid_test200"] = hyb_test
    save_summary(sel, sync_root)

    # 2) 限角 la120 M2 下界复核：补测 zeta in {0.05, 0.1, 0.25}（原网格 {0.5, 1.0, 2.0} 中 0.5 处为 25.87 dB）
    la_val = {}
    for z in (0.05, 0.1, 0.25):
        out = RESULTS / f"tune_la120_pigdm_{z:g}"
        run_eval(sample_cmd(args.prior, "la120", "pigdm", out, args.data_dir, args.eval_bs,
                            split="val", n_test=N_TUNE, zeta=z), out, sync_root)
        la_val[str(z)] = {"zeta": z, "val_psnr": psnr_of(out)}
    # 同时记录原网格 0.5 作为参照（如本地无则重跑一次 val 确认）
    out_05 = RESULTS / "tune_la120_pigdm_0.5"
    run_eval(sample_cmd(args.prior, "la120", "pigdm", out_05, args.data_dir, args.eval_bs,
                        split="val", n_test=N_TUNE, zeta=0.5), out_05, sync_root)
    la_val["0.5"] = {"zeta": 0.5, "val_psnr": psnr_of(out_05)}
    best_la_z = max(la_val.values(), key=lambda d: d["val_psnr"])["zeta"]
    t12["la120_m2_val"] = la_val
    t12["la120_m2_best_zeta"] = best_la_z
    save_summary(sel, sync_root)

    # 若下界补测找到了比 0.5 更高的 val PSNR，则立刻在 test200 上更新 la120 与 la90 的 M2 成绩
    if best_la_z != 0.5:
        la_test = {}
        for s in ("la120", "la90"):
            name = f"main_{s}_pigdm_z{best_la_z:g}"
            out = RESULTS / name
            run_eval(sample_cmd(args.prior, s, "pigdm", out, args.data_dir, args.eval_bs, zeta=best_la_z),
                     out, sync_root)
            la_test[name] = {"psnr": psnr_of(out), "minutes": minutes_of(out)}
        t12["la_m2_updated_test200"] = la_test
        save_summary(sel, sync_root)
    print("\n=== T1-2 SUMMARY ===", json.dumps(t12, indent=2), flush=True)


def stage_t1_3_grad(args, sync_root: Path | None):
    """T1-3 ③: 最终 120k 权重复测 M1 梯度曲线与 t_start=0 上限。"""
    out = RESULTS / "m1_grad_120k.json"
    if out.exists():
        print(f"[SKIP] {out}", flush=True)
        return
    cmd = ["scripts/measure_m1_grad.py", "--ckpt", args.prior, "--config", SETTINGS["sv60"],
           "--data-dir", args.data_dir, "--out", str(out)]
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run([sys.executable, "-u", *cmd], check=True)
    if sync_root:
        (sync_root / "results").mkdir(parents=True, exist_ok=True)
        shutil.copy2(out, sync_root / "results" / out.name)


def _train_fbpconvnet(cfg_path: str, name: str, max_steps: int, args, sync_root: Path | None):
    dst = RESULTS / name
    if (dst / "metrics.json").exists():
        print(f"[SKIP] {dst}", flush=True)
        return
    ckpt_dir = Path("checkpoints") / name
    sync_ckpt = sync_root / "checkpoints" / name if sync_root else None
    if sync_ckpt and (sync_ckpt / "ckpt_last.pt").exists() and not (ckpt_dir / "ckpt_last.pt").exists():
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sync_ckpt / "ckpt_last.pt", ckpt_dir / "ckpt_last.pt")
        print(f"[RESTORE] 从 Drive 恢复 {name} 断点", flush=True)
    cmd = ["src/baselines/train_fbpconvnet.py", "--config", cfg_path,
           "--out-dir", str(ckpt_dir), "--max-steps", str(max_steps), "--resume", "--save-every", "2000"]
    if args.data_dir:
        cmd += ["--data-dir", args.data_dir]
    if sync_ckpt:
        cmd += ["--sync-dir", str(sync_ckpt)]
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run([sys.executable, "-u", *cmd], check=True)
    dst.mkdir(parents=True, exist_ok=True)
    for fn in ("metrics.json", "per_image.csv", "samples.png", "first4.npy"):
        shutil.copy2(ckpt_dir / fn, dst / fn)
    shutil.copy2(cfg_path, dst / "config.yaml")
    sync_dir_tree(dst, sync_root / "results" / name if sync_root else None)


def stage_t1_3_fbp_la(args, sync_root: Path | None):
    """T1-3 ①: 补训 fbpconvnet_la120 / fbpconvnet_la90（各 10k 步，与主表其余 4 协议一致）。"""
    _train_fbpconvnet("configs/fbpconvnet_la120.yaml", "fbpconvnet_la120", 10000, args, sync_root)
    _train_fbpconvnet("configs/fbpconvnet_la90.yaml", "fbpconvnet_la90", 10000, args, sync_root)


def stage_t1_3_fbp_30k(args, sync_root: Path | None):
    """T1-3 ②: sv60 与 ld1e5 各跑 30k 步长训 FBPConvNet（约 11.1 epoch）。"""
    _train_fbpconvnet("configs/fbpconvnet_sv60.yaml", "fbpconvnet_sv60_30k", 30000, args, sync_root)
    _train_fbpconvnet("configs/fbpconvnet_ld1e5.yaml", "fbpconvnet_ld1e5_30k", 30000, args, sync_root)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--sync-dir", default=None)
    ap.add_argument("--eval-bs", type=int, default=16)
    ap.add_argument("--stage", nargs="+", required=True,
                    choices=["t1_1", "t1_2", "t1_3_grad", "t1_3_fbp_la", "t1_3_fbp_30k"])
    args = ap.parse_args()
    sync_root = Path(args.sync_dir) if args.sync_dir else None

    for st in args.stage:
        print(f"\n[STAGE] {st}", flush=True)
        if st == "t1_1":
            stage_t1_1(args, sync_root)
        elif st == "t1_2":
            stage_t1_2(args, sync_root)
        elif st == "t1_3_grad":
            stage_t1_3_grad(args, sync_root)
        elif st == "t1_3_fbp_la":
            stage_t1_3_fbp_la(args, sync_root)
        elif st == "t1_3_fbp_30k":
            stage_t1_3_fbp_30k(args, sync_root)


if __name__ == "__main__":
    main()
