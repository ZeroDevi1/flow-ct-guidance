"""Colab GPU 选卡性价比基准测试（约 3–5 分钟）：测训练步速、显存与 dcfull 推理耗时。

用法（在 Colab GPU 会话中，仓库 code/ 目录下）：
    python scripts/bench_colab_gpu.py --prior checkpoints/prior_256/ema_final.pt --data-dir /content/data/processed
"""
from __future__ import annotations

import argparse
import json
import random
import time

import torch

from src.data.dataset import SliceDataset, SyntheticPhantomDataset
from src.models.angle_ssm import AngleCompleter
from src.physics.radon import CTOperator
from src.sample_posterior import build_full_context, posterior_sample
from src.train_completer import loss_fn, make_inputs, sample_mask
from src.utils import load_ema, set_seed


def bench_train(prior, op, ds, bs: int, warmup: int, steps: int, device: str):
    model = AngleCompleter(n_det=256, d=256, n_layers=4, mixer="ssm", t_cond=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    rng = random.Random(0)
    x = torch.stack([ds[i % len(ds)] for i in range(bs)]).to(device)
    torch.cuda.reset_peak_memory_stats(device)
    t0 = None
    for i in range(warmup + steps):
        if i == warmup:
            torch.cuda.synchronize(device)
            t0 = time.time()
        mask, _ = sample_mask(rng, 0.4)
        mask = mask.to(device)
        t = torch.rand(bs, device=device)
        s_true, y_full, s_pred = make_inputs(prior, op, x, mask, t, p_dc=0.5, dc_lam=0.3, use_amp=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y_hat, lv = model(s_pred, y_full, mask, t)
        loss, _ = loss_fn(y_hat.float(), lv.float(), s_true, y_full, mask, beta_hlcc=0.1, nll_weight=0.01)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    torch.cuda.synchronize(device)
    dt = time.time() - t0
    sec_per_step = dt / steps
    peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    return {
        "batch_size": bs,
        "sec_per_step": round(sec_per_step, 4),
        "slices_per_sec": round(bs / sec_per_step, 2),
        "est_hours_15k": round(15000 * sec_per_step / 3600, 2),
        "peak_vram_gb": round(peak_gb, 2),
    }


@torch.no_grad()
def bench_eval(prior, op_full, ds, bs: int, n_imgs: int, device: str):
    completer = AngleCompleter(n_det=256, d=256, n_layers=4, mixer="ssm", t_cond=True).to(device).eval()
    cfg = {"physics": {"n_angles": 60, "angle_range": 3.141592653589793}}
    op60 = CTOperator(256, n_angles=60, device=device)
    torch.cuda.synchronize(device)
    t0 = time.time()
    for i0 in range(0, n_imgs, bs):
        x = torch.stack([ds[(i0 + j) % len(ds)] for j in range(min(bs, n_imgs - i0))]).to(device)
        y = op60(x)
        full = build_full_context(cfg, 256, device, y, "ssm", completer, kappa=0.3,
                                  use_conf=True, use_hlcc=True, op_full=op_full)
        _ = posterior_sample(prior, op60, y, x.shape, "dcfull", nfe=50, dc_lam=0.3, dc_iters=5, full=full)
    torch.cuda.synchronize(device)
    dt = time.time() - t0
    return {
        "eval_bs": bs,
        "n_imgs": n_imgs,
        "sec_total": round(dt, 2),
        "est_min_test200": round(dt * (200 / n_imgs) / 60, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", required=True)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--no-compile", action="store_true")
    args = ap.parse_args()

    set_seed(0)
    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"[GPU] {gpu_name} ({vram_gb:.1f} GB VRAM)", flush=True)

    prior, _ = load_ema(args.prior, device)
    if not args.no_compile:
        prior = torch.compile(prior)
    op = CTOperator(256, n_angles=180, device=device)
    ds = SliceDataset(args.data_dir, "val", augment=False, max_items=64, subsample="even") if args.data_dir else \
        SyntheticPhantomDataset(n=64, size=256)

    train_res = []
    for bs in args.batch_sizes:
        try:
            r = bench_train(prior, op, ds, bs, args.warmup, args.steps, device)
            print("[TRAIN]", json.dumps(r), flush=True)
            train_res.append(r)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"[TRAIN] bs={bs} OOM", flush=True)

    eval_res = bench_eval(prior, op, ds, bs=min(16, max(args.batch_sizes)), n_imgs=32, device=device)
    print("[EVAL]", json.dumps(eval_res), flush=True)
    print("[SUMMARY]", json.dumps({"gpu": gpu_name, "vram_gb": round(vram_gb, 1),
                                    "train": train_res, "eval": eval_res}, indent=2), flush=True)


if __name__ == "__main__":
    main()
