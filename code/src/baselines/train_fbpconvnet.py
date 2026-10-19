"""B1：FBPConvNet —— 监督学习基线。输入 FBP 重建，U-Net 预测残差，每种采样/剂量设置单独训练。

用法：
    python src/baselines/train_fbpconvnet.py --config configs/fbpconvnet_sv60.yaml
训练结束自动在测试集上评测并写 metrics.json。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import SliceDataset, SyntheticPhantomDataset
from src.eval.metrics import data_consistency, lpips_dist, psnr, ssim, summarize
from src.models.unet import UNet, count_params
from src.sample_posterior import build_operator, make_measurement
from src.train_flow import lr_at
from src.utils import save_first, save_grid, save_per_image, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.max_steps:
        cfg["train"]["max_steps"] = args.max_steps
    set_seed(cfg.get("seed", 0))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(cfg["out_dir"]); out.mkdir(parents=True, exist_ok=True)
    n = cfg["model"]["image_size"]
    op = build_operator(cfg, n, device)

    if cfg["data"]["type"] == "phantom":
        tr = SyntheticPhantomDataset(n=cfg["data"].get("n", 512), size=n)
        te = SyntheticPhantomDataset(n=cfg["data"].get("n_test", 32), size=n, seed=123)
    else:
        tr = SliceDataset(cfg["data"]["processed_dir"], "train", augment=True, max_items=cfg["data"].get("max_items"))
        te = SliceDataset(cfg["data"]["processed_dir"], cfg["data"].get("test_split", "test"), augment=False,
                          max_items=cfg["data"].get("n_test", 200))
    dl = DataLoader(tr, batch_size=cfg["train"]["batch_size"], shuffle=True, drop_last=True,
                    num_workers=cfg["train"].get("num_workers", 4))
    model = UNet(**cfg["model"]).to(device)
    print(f"params {count_params(model):.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])
    gen = torch.Generator(device=device).manual_seed(1)
    step, t0 = 0, time.time()
    tzero = torch.zeros(cfg["train"]["batch_size"], device=device)
    model.train()
    while step < cfg["train"]["max_steps"]:
        for x in dl:
            x = x.to(device)
            # 与先验相同的 warmup + 余弦衰减（lr_min 未设时为常数学习率）
            for g in opt.param_groups:
                g["lr"] = lr_at(step, cfg["train"])
            with torch.no_grad():
                y = make_measurement(op, x, cfg, gen)
                x_fbp = op.fbp(y)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                pred = x_fbp + model(x_fbp, tzero[: x.shape[0]]).float()
            loss = ((pred - x) ** 2).mean()
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            step += 1
            if step % 100 == 0:
                print(json.dumps({"step": step, "loss": float(loss.detach()), "min": (time.time() - t0) / 60}))
            if step >= cfg["train"]["max_steps"]:
                break
    torch.save({"model": model.state_dict(), "cfg": cfg}, out / "model.pt")

    model.eval(); agg = {"psnr": [], "ssim": [], "lpips": [], "dc": []}; first = None
    with torch.no_grad():
        for i0 in range(0, len(te), 16):
            idx = list(range(i0, min(i0 + 16, len(te))))
            x = torch.stack([te[j] for j in idx]).to(device)
            y = make_measurement(op, x, cfg, indices=idx)
            x_fbp = op.fbp(y)
            pred = (x_fbp + model(x_fbp, torch.zeros(x.shape[0], device=device))).clamp(-1, 1)
            agg["psnr"] += psnr(pred, x, op.mask).tolist(); agg["ssim"] += ssim(pred, x, op.mask).tolist()
            agg["lpips"] += lpips_dist(pred, x).tolist()
            agg["dc"] += data_consistency(op, pred, y).tolist()
            if first is None:
                first = (x[:4], x_fbp[:4], pred[:4])
    res = {"method": "fbpconvnet", "physics": cfg["physics"], "metrics": summarize(agg), "steps": step,
           "minutes": (time.time() - t0) / 60}
    json.dump(res, open(out / "metrics.json", "w"), indent=2)
    save_grid(torch.cat(first), out / "samples.png", nrow=4)
    save_first(out, list(first)); save_per_image(out, agg)
    print(json.dumps(res["metrics"], indent=1))


if __name__ == "__main__":
    main()
