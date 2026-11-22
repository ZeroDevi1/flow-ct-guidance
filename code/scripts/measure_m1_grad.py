"""T1-3 ③：使用最终 120k 步先验权重（ema_final.pt）复测 M1（DPS）引导梯度随 t 的变化曲线。

输出 JSON 包含：
- unguided 轨迹上每步 t_i 的标准化 M1 梯度 RMS、原始梯度 RMS、速度场 RMS、相对数据残差；
- guided（t_start=0.4, zeta=30）轨迹上的同样统计；
- t_start=0.0 时在 val（50 张）上的快速网格检验（zeta in {1, 3, 10, 30}），用于确认从 t=0 施加引导的上限。

用法：
    python scripts/measure_m1_grad.py --ckpt checkpoints/prior_256/ema_final.pt \
        --config configs/eval_sparse60.yaml --data-dir /content/data/processed \
        --out results_t1/m1_grad_120k.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from src.data.dataset import SliceDataset
from src.eval.metrics import psnr
from src.sample_posterior import build_operator, make_measurement, posterior_sample
from src.utils import load_ema, set_seed


def trace_m1_gradients(model, op, x_gt, y, nfe: int, zeta: float = 0.0, t_start: float = 1.0, seed: int = 0):
    """沿 ODE 轨迹记录每一步的 M1 梯度统计。zeta=0 表示无引导先验轨迹。"""
    device = y.device
    b = x_gt.shape[0]
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(x_gt.shape, device=device, generator=gen)
    ts = torch.linspace(0, 1, nfe + 1, device=device)
    s_sqrt = op.ata_scale ** 0.5
    y_norm = y.flatten(1).norm(dim=1).clamp(min=1e-8)

    steps = []
    for i in range(nfe):
        t0, t1 = ts[i], ts[i + 1]
        dt = t1 - t0
        tb = t0.expand(b)
        with torch.enable_grad():
            x_req = x.detach().requires_grad_(True)
            v = model(x_req, tb)
            x1 = x_req + (1 - t0) * v
            r = op(x1) - y
            norm = r.flatten(1).norm(dim=1)
            g_raw, = torch.autograd.grad(norm.sum(), x_req)
        g_normed = g_raw / s_sqrt
        grad_rms = float(g_normed.pow(2).flatten(1).mean(1).sqrt().mean())
        raw_rms = float(g_raw.pow(2).flatten(1).mean(1).sqrt().mean())
        vel_rms = float(v.detach().pow(2).flatten(1).mean(1).sqrt().mean())
        rel_res = float((norm.detach() / y_norm).mean())
        steps.append({
            "step": i,
            "t": round(float(t0), 4),
            "grad_rms": grad_rms,
            "raw_grad_rms": raw_rms,
            "vel_rms": vel_rms,
            "ratio_grad_vel": grad_rms / max(vel_rms, 1e-12),
            "rel_residual": rel_res,
        })
        with torch.no_grad():
            if zeta > 0 and float(t0) >= t_start:
                x = (x_req + dt * (v - zeta * g_normed)).detach()
            else:
                x = (x_req + dt * v).detach()
    final_psnr = float(psnr(x.clamp(-1, 1), x_gt, op.mask).mean())
    return {"final_psnr": final_psnr, "steps": steps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="configs/eval_sparse60.yaml")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--n-trace", type=int, default=16, help="用于梯度轨迹统计的 val 切片数")
    ap.add_argument("--n-ts0", type=int, default=50, help="用于 t_start=0 上限验证的 val 切片数")
    ap.add_argument("--nfe", type=int, default=50)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    cfg["data"]["processed_dir"] = args.data_dir
    set_seed(cfg.get("seed", 0))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, mcfg = load_ema(args.ckpt, device)
    n = mcfg["model"]["image_size"]
    op = build_operator(cfg, n, device)

    ds_trace = SliceDataset(args.data_dir, split="val", augment=False, max_items=args.n_trace, subsample="even")
    idx = list(range(len(ds_trace)))
    x_gt = torch.stack([ds_trace[j] for j in idx]).to(device)
    y = make_measurement(op, x_gt, cfg, indices=idx)

    unguided = trace_m1_gradients(model, op, x_gt, y, nfe=args.nfe, zeta=0.0, t_start=1.0, seed=cfg.get("seed", 0))
    guided_ts04 = trace_m1_gradients(model, op, x_gt, y, nfe=args.nfe, zeta=30.0, t_start=0.4, seed=cfg.get("seed", 0))
    guided_ts00 = trace_m1_gradients(model, op, x_gt, y, nfe=args.nfe, zeta=10.0, t_start=0.0, seed=cfg.get("seed", 0))

    # 摘要：t=0 与 t>=0.2 的峰值/均值比
    u_steps = unguided["steps"]
    g0 = u_steps[0]["grad_rms"]
    late = [s["grad_rms"] for s in u_steps if s["t"] >= 0.2]
    summary = {
        "t0_grad_rms": g0,
        "t_ge_02_min": min(late),
        "t_ge_02_max": max(late),
        "t_ge_02_mean": sum(late) / len(late),
        "spike_ratio_vs_mean": g0 / max(sum(late) / len(late), 1e-12),
        "spike_ratio_vs_min": g0 / max(min(late), 1e-12),
    }

    # 用最终 120k 权重在 val (50 张) 上复测 t_start=0.0 的最高 PSNR
    ds_val = SliceDataset(args.data_dir, split="val", augment=False, max_items=args.n_ts0, subsample="even")
    ts0_grid = {}
    for z in (1.0, 3.0, 10.0, 30.0):
        gen = torch.Generator(device=device).manual_seed(cfg.get("seed", 0))
        ps = []
        for i0 in range(0, len(ds_val), 16):
            bidx = list(range(i0, min(i0 + 16, len(ds_val))))
            xb = torch.stack([ds_val[j] for j in bidx]).to(device)
            yb = make_measurement(op, xb, cfg, indices=bidx)
            xh = posterior_sample(model, op, yb, xb.shape, "dps", args.nfe, zeta=z,
                                  generator=gen, t_start_guidance=0.0)
            ps += psnr(xh, xb, op.mask).tolist()
        ts0_grid[str(z)] = sum(ps) / len(ps)
        print(f"[ts0] zeta={z:g} val_psnr={ts0_grid[str(z)]:.2f}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ckpt": args.ckpt,
        "config": args.config,
        "n_trace": len(ds_trace),
        "n_ts0": len(ds_val),
        "nfe": args.nfe,
        "summary_unguided": summary,
        "ts0_val_psnr_grid": ts0_grid,
        "ts0_best_psnr": max(ts0_grid.values()),
        "unguided": unguided,
        "guided_ts04_z30": guided_ts04,
        "guided_ts00_z10": guided_ts00,
    }
    out.write_text(json.dumps(payload, indent=2))
    print("[SUMMARY]", json.dumps(summary, indent=2), flush=True)
    print("[TS0]", json.dumps(ts0_grid, indent=2), flush=True)
    print("->", out, flush=True)


if __name__ == "__main__":
    main()
