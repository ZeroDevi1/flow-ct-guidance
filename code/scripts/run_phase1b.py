"""Phase 1b 实验矩阵编排器（支持断点续跑、Google Drive 实时同步与止损门控报告）。

阶段（--stage，可按序传入多个）：
    baselines        限角 la120/la90 的 FBP + M2/M3 调参与主表；5 设置不学习对照 dcfull-interp/pred 调参与主表
    train_ssm        训练主补全器 completer_ssm（自动 --resume 并同步权重到 Drive）
    gate             在 val 上为 dcfull(ssm) 选 κ，在 test200 跑 sv60/la120（含/不含 HLCC），输出止损判定表
    ablations_train  训练 4 个消融补全器：conv / attn / ssm_not / ssm_nohlcc
    main_full        补齐 5 设置主表、全部消融评测、NFE 消融（20）与 K=8 不确定性（sv60/la120）
    fbpconvnet_la    训练限角 la120/la90 的监督基线 FBPConvNet

用法：
    python scripts/run_phase1b.py --prior checkpoints/prior_256/ema_final.pt \
        --data-dir /content/data/processed --sync-dir /content/drive/MyDrive/flow-ct-guidance/phase1b \
        --stage baselines train_ssm gate
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
}
KAPPA_GRID = (0.05, 0.15, 0.3, 0.6, 1.0)
M2_GRID = (0.5, 1.0, 2.0)
M3_GRID = (0.1, 0.3, 1.0, 3.0)
N_TUNE = 50
RESULTS = Path("results_p1b")


def sync_dir_tree(src: Path, dst: Path | None):
    if dst is None or not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        if p.is_file() and p.suffix in (".json", ".csv", ".yaml", ".png", ".pt"):
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
               full_mode: str | None = None, completer: str | None = None,
               kappa: float | None = None, no_conf: bool = False, hlcc: bool = False,
               hlcc_nmax: int = 6) -> list[str]:
    cmd = ["src/sample_posterior.py", "--ckpt", prior, "--config", SETTINGS[setting],
           "--guidance", guidance, "--nfe", str(nfe), "--n-samples", str(n_samples), "--out", str(out)]
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
    if full_mode:
        cmd += ["--full-mode", full_mode]
    if completer:
        cmd += ["--completer", completer]
    if kappa is not None:
        cmd += ["--kappa", str(kappa)]
    if no_conf:
        cmd += ["--no-conf"]
    if hlcc:
        cmd += ["--hlcc", "--hlcc-nmax", str(hlcc_nmax)]
    return cmd


def psnr_of(out: Path) -> float:
    return float(json.loads((out / "metrics.json").read_text())["metrics"]["psnr"]["mean"])


def best_param(scores: dict[float, float]) -> float:
    return max(scores, key=scores.get)


def load_or_init_params() -> dict:
    p = RESULTS / "selected_p1b.json"
    return json.loads(p.read_text()) if p.exists() else {}


def save_params(params: dict, sync_root: Path | None):
    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / "selected_p1b.json"
    p.write_text(json.dumps(params, indent=2))
    if sync_root:
        (sync_root / "results").mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, sync_root / "results" / p.name)


def train_completer(cfg_path: str, name: str, args):
    out = Path("checkpoints") / name
    final_step_ckpt = out / f"completer_step{args.max_steps}.pt"
    if final_step_ckpt.exists():
        print(f"[SKIP] {name} 已完成 ({final_step_ckpt})", flush=True)
        return str(out / "completer.pt")
    sync_ckpt = Path(args.sync_dir) / "checkpoints" / name if args.sync_dir else None
    # 若 Drive 有历史断点而本地没有，先拉回本地续训
    if sync_ckpt and (sync_ckpt / "ckpt_last.pt").exists() and not (out / "ckpt_last.pt").exists():
        out.mkdir(parents=True, exist_ok=True)
        for f in sync_ckpt.iterdir():
            if f.is_file():
                shutil.copy2(f, out / f.name)
        print(f"[RESTORE] 从 Drive 恢复 {name} 断点", flush=True)
    cmd = ["src/train_completer.py", "--config", cfg_path, "--prior-ckpt", args.prior,
           "--out-dir", str(out), "--max-steps", str(args.max_steps), "--resume"]
    if args.data_dir:
        cmd += ["--data-dir", args.data_dir]
    if args.train_bs:
        cmd += ["--batch-size", str(args.train_bs)]
    if sync_ckpt:
        cmd += ["--sync-dir", str(sync_ckpt)]
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run([sys.executable, "-u", *cmd], check=True)
    return str(out / "completer.pt")


def stage_baselines(args, sync_root: Path | None):
    params = load_or_init_params()
    # 1. 限角 FBP
    for s in ("la120", "la90"):
        out = RESULTS / f"fbp_{s}"
        cmd = ["src/baselines/fbp_eval.py", "--config", SETTINGS[s], "--out", str(out)]
        run_eval(cmd, out, sync_root)

    # 2. 限角 la120 上为 M2(pigdm) / M3(dc) 选参，并跑 la120 + la90 主表
    m2_scores = {}
    for z in M2_GRID:
        out = RESULTS / f"tune_la120_pigdm_{z:g}"
        run_eval(sample_cmd(args.prior, "la120", "pigdm", out, args.data_dir, args.eval_bs,
                            split="val", n_test=N_TUNE, zeta=z), out, sync_root)
        m2_scores[z] = psnr_of(out)
    best_m2 = best_param(m2_scores)

    m3_scores = {}
    for lam in M3_GRID:
        out = RESULTS / f"tune_la120_dc_{lam:g}"
        run_eval(sample_cmd(args.prior, "la120", "dc", out, args.data_dir, args.eval_bs,
                            split="val", n_test=N_TUNE, dc_lam=lam), out, sync_root)
        m3_scores[lam] = psnr_of(out)
    best_m3 = best_param(m3_scores)

    params["la_m2_zeta"] = best_m2
    params["la_m3_lam"] = best_m3
    params.setdefault("sv_m3_lam", 0.3)  # Phase 1 sv60 选定值
    save_params(params, sync_root)

    for s in ("la120", "la90"):
        run_eval(sample_cmd(args.prior, s, "pigdm", RESULTS / f"main_{s}_pigdm", args.data_dir,
                            args.eval_bs, zeta=best_m2), RESULTS / f"main_{s}_pigdm", sync_root)
        run_eval(sample_cmd(args.prior, s, "dc", RESULTS / f"main_{s}_dc", args.data_dir,
                            args.eval_bs, dc_lam=best_m3), RESULTS / f"main_{s}_dc", sync_root)

    # 3. 不学习的 dcfull 对照：interp / pred，在 sv60 和 la120 上选 κ，再跑 5 设置主表
    for mode in ("interp", "pred"):
        for anchor, group in (("sv60", ("sv30", "sv60", "sv90")), ("la120", ("la120", "la90"))):
            lam = params["sv_m3_lam"] if anchor == "sv60" else params["la_m3_lam"]
            k_scores = {}
            for k in KAPPA_GRID:
                out = RESULTS / f"tune_{anchor}_dcfull_{mode}_k{k:g}"
                run_eval(sample_cmd(args.prior, anchor, "dcfull", out, args.data_dir, args.eval_bs,
                                    split="val", n_test=N_TUNE, dc_lam=lam, full_mode=mode, kappa=k),
                         out, sync_root)
                k_scores[k] = psnr_of(out)
            best_k = best_param(k_scores)
            params[f"kappa_{mode}_{anchor}"] = best_k
            save_params(params, sync_root)
            for s in group:
                out = RESULTS / f"main_{s}_dcfull_{mode}"
                run_eval(sample_cmd(args.prior, s, "dcfull", out, args.data_dir, args.eval_bs,
                                    dc_lam=lam, full_mode=mode, kappa=best_k), out, sync_root)


def stage_gate(args, sync_root: Path | None):
    params = load_or_init_params()
    params.setdefault("sv_m3_lam", 0.3)
    params.setdefault("la_m3_lam", 0.3)
    ckpt = str(Path("checkpoints/completer_ssm/completer.pt"))
    if not Path(ckpt).exists():
        raise SystemExit(f"缺少主补全器权重 {ckpt}，请先执行 --stage train_ssm")

    report = {}
    for anchor in ("sv60", "la120"):
        lam = params["sv_m3_lam"] if anchor == "sv60" else params["la_m3_lam"]
        k_scores = {}
        for k in KAPPA_GRID:
            out = RESULTS / f"tune_{anchor}_dcfull_ssm_k{k:g}"
            run_eval(sample_cmd(args.prior, anchor, "dcfull", out, args.data_dir, args.eval_bs,
                                split="val", n_test=N_TUNE, dc_lam=lam, full_mode="ssm",
                                completer=ckpt, kappa=k), out, sync_root)
            k_scores[k] = psnr_of(out)
        best_k = best_param(k_scores)
        params[f"kappa_ssm_{anchor}"] = best_k
        save_params(params, sync_root)

        out_plain = RESULTS / f"main_{anchor}_dcfull_ssm"
        out_hlcc = RESULTS / f"main_{anchor}_dcfull_ssm_hlcc"
        run_eval(sample_cmd(args.prior, anchor, "dcfull", out_plain, args.data_dir, args.eval_bs,
                            dc_lam=lam, full_mode="ssm", completer=ckpt, kappa=best_k, hlcc=False),
                 out_plain, sync_root)
        run_eval(sample_cmd(args.prior, anchor, "dcfull", out_hlcc, args.data_dir, args.eval_bs,
                            dc_lam=lam, full_mode="ssm", completer=ckpt, kappa=best_k, hlcc=True,
                            hlcc_nmax=args.hlcc_nmax),
                 out_hlcc, sync_root)

        # 读取参照基线（sv60 的 M3 若不在 results_p1b 则读仓库已有 results/main_sv60_dc）
        m3_path = RESULTS / f"main_{anchor}_dc"
        if not (m3_path / "metrics.json").exists():
            m3_path = Path("../results") / f"main_{anchor}_dc"
        interp_path = RESULTS / f"main_{anchor}_dcfull_interp"
        m3_psnr = psnr_of(m3_path) if (m3_path / "metrics.json").exists() else None
        interp_psnr = psnr_of(interp_path) if (interp_path / "metrics.json").exists() else None
        ssm_psnr = psnr_of(out_plain)
        hlcc_psnr = psnr_of(out_hlcc)
        report[anchor] = {
            "best_kappa": best_k,
            "M3_dc": m3_psnr,
            "dcfull_interp": interp_psnr,
            "dcfull_ssm": ssm_psnr,
            "dcfull_ssm_hlcc": hlcc_psnr,
            "gain_vs_M3": round(max(ssm_psnr, hlcc_psnr) - m3_psnr, 3) if m3_psnr is not None else None,
        }
    print("\n=== PHASE 1B STOP-LOSS GATE REPORT ===", flush=True)
    print(json.dumps(report, indent=2), flush=True)
    sv_gain = report["sv60"]["gain_vs_M3"]
    la_gain = report["la120"]["gain_vs_M3"]
    passed = (sv_gain is not None and sv_gain >= 0.3) or (la_gain is not None and la_gain >= 0.5)
    print(f"[GATE] sv60_gain={sv_gain} dB, la120_gain={la_gain} dB -> {'PASS' if passed else 'STOP_LOSS_TRIGGERED'}",
          flush=True)


def stage_ablations_train(args):
    for cfg, name in (
        ("configs/completer_conv.yaml", "completer_conv"),
        ("configs/completer_attn.yaml", "completer_attn"),
        ("configs/completer_ssm_not.yaml", "completer_ssm_not"),
        ("configs/completer_ssm_nohlcc.yaml", "completer_ssm_nohlcc"),
    ):
        train_completer(cfg, name, args)


def stage_main_full(args, sync_root: Path | None):
    params = load_or_init_params()
    ssm_ckpt = "checkpoints/completer_ssm/completer.pt"
    # 1. 补齐剩余 3 个设置主表
    for s, anchor in (("sv30", "sv60"), ("sv90", "sv60"), ("la90", "la120")):
        lam = params.get("sv_m3_lam", 0.3) if anchor == "sv60" else params.get("la_m3_lam", 0.3)
        k = params[f"kappa_ssm_{anchor}"]
        for hlcc, suf in ((False, ""), (True, "_hlcc")):
            out = RESULTS / f"main_{s}_dcfull_ssm{suf}"
            run_eval(sample_cmd(args.prior, s, "dcfull", out, args.data_dir, args.eval_bs,
                                dc_lam=lam, full_mode="ssm", completer=ssm_ckpt, kappa=k,
                                hlcc=hlcc, hlcc_nmax=args.hlcc_nmax), out, sync_root)

    # 2. 消融表（sv60 + la120）：conv / attn / ssm_not / ssm_nohlcc / ssm(no-conf)
    for anchor in ("sv60", "la120"):
        lam = params.get("sv_m3_lam", 0.3) if anchor == "sv60" else params.get("la_m3_lam", 0.3)
        k = params[f"kappa_ssm_{anchor}"]
        for tag in ("conv", "attn", "ssm_not", "ssm_nohlcc"):
            ckpt = f"checkpoints/completer_{tag}/completer.pt"
            if not Path(ckpt).exists():
                print(f"[WARN] 跳过缺失权重 {ckpt}", flush=True)
                continue
            out = RESULTS / f"ablation_{anchor}_{tag}"
            run_eval(sample_cmd(args.prior, anchor, "dcfull", out, args.data_dir, args.eval_bs,
                                dc_lam=lam, full_mode="ssm", completer=ckpt, kappa=k), out, sync_root)
        out_nc = RESULTS / f"ablation_{anchor}_noconf"
        run_eval(sample_cmd(args.prior, anchor, "dcfull", out_nc, args.data_dir, args.eval_bs,
                            dc_lam=lam, full_mode="ssm", completer=ssm_ckpt, kappa=k, no_conf=True),
                 out_nc, sync_root)

        # 3. NFE 20 与 K=8 后验均值
        out_nfe20 = RESULTS / f"nfe20_{anchor}_dcfull_ssm"
        run_eval(sample_cmd(args.prior, anchor, "dcfull", out_nfe20, args.data_dir, args.eval_bs,
                            nfe=20, dc_lam=lam, full_mode="ssm", completer=ssm_ckpt, kappa=k),
                 out_nfe20, sync_root)
        out_unc8 = RESULTS / f"unc8_{anchor}_dcfull_ssm"
        run_eval(sample_cmd(args.prior, anchor, "dcfull", out_unc8, args.data_dir, args.eval_bs,
                            nfe=50, n_samples=8, dc_lam=lam, full_mode="ssm", completer=ssm_ckpt, kappa=k),
                 out_unc8, sync_root)


def stage_fbpconvnet_la(args, sync_root: Path | None):
    for s in ("la120", "la90"):
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
        sync_dir_tree(dst, sync_root / "results" / dst.name if sync_root else None)
        if sync_root:
            (sync_root / "checkpoints").mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_dir / "model.pt", sync_root / "checkpoints" / f"fbpconvnet_{s}.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", required=True)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--sync-dir", default=None)
    ap.add_argument("--train-bs", type=int, default=16)
    ap.add_argument("--eval-bs", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=15000)
    ap.add_argument("--hlcc-nmax", type=int, default=6)
    ap.add_argument("--stage", nargs="+", required=True,
                    choices=["baselines", "train_ssm", "gate", "ablations_train", "main_full", "fbpconvnet_la"])
    args = ap.parse_args()
    sync_root = Path(args.sync_dir) if args.sync_dir else None

    for st in args.stage:
        print(f"\n[STAGE] {st}", flush=True)
        if st == "baselines":
            stage_baselines(args, sync_root)
        elif st == "train_ssm":
            train_completer("configs/completer_ssm.yaml", "completer_ssm", args)
        elif st == "gate":
            stage_gate(args, sync_root)
        elif st == "ablations_train":
            stage_ablations_train(args)
        elif st == "main_full":
            stage_main_full(args, sync_root)
        elif st == "fbpconvnet_la":
            stage_fbpconvnet_la(args, sync_root)


if __name__ == "__main__":
    main()
