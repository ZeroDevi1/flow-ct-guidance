"""B1：FBPConvNet —— 监督学习基线。输入 FBP 重建，U-Net 预测残差，每种采样/剂量设置单独训练。

用法：
    python src/baselines/train_fbpconvnet.py --config configs/fbpconvnet_sv60.yaml
训练结束自动在测试集上评测并写 metrics.json。
"""
from __future__ import annotations

import argparse
import json
import shutil
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


def _sync_file(src: Path, sync_dir: Path | None):
    if sync_dir is None or not src.exists():
        return
    sync_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, sync_dir / src.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--data-dir", default=None, help="覆盖 data.processed_dir")
    ap.add_argument("--out-dir", default=None, help="覆盖 out_dir")
    ap.add_argument("--batch-size", type=int, default=None, help="覆盖 train.batch_size")
    ap.add_argument("--resume", action="store_true", help="若 out_dir/ckpt_last.pt 存在则续训")
    ap.add_argument("--save-every", type=int, default=2000, help="每隔多少步保存一次 ckpt_last.pt")
    ap.add_argument("--sync-dir", default=None, help="可选：定期将断点与最终产物同步到此目录（如 Drive）")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.max_steps:
        cfg["train"]["max_steps"] = args.max_steps
    if args.data_dir:
        cfg["data"]["processed_dir"] = args.data_dir
    if args.out_dir:
        cfg["out_dir"] = args.out_dir
    if args.batch_size:
        cfg["train"]["batch_size"] = args.batch_size
    sync_dir = Path(args.sync_dir) if args.sync_dir else None
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
    nw = cfg["train"].get("num_workers", 4)
    dl = DataLoader(tr, batch_size=cfg["train"]["batch_size"], shuffle=True, drop_last=True,
                    num_workers=nw, pin_memory=(device == "cuda"), persistent_workers=(nw > 0))
    model = UNet(**cfg["model"]).to(device)
    print(f"params {count_params(model):.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])
    gen = torch.Generator(device=device).manual_seed(1)
    step, t0 = 0, time.time()
    ckpt_last = out / "ckpt_last.pt"
    if args.resume and ckpt_last.exists():
        ck = torch.load(ckpt_last, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step = int(ck["step"])
        print(f"[RESUME] {ckpt_last} step={step}/{cfg['train']['max_steps']}", flush=True)
    tzero = torch.zeros(cfg["train"]["batch_size"], device=device)
    model.train()
    while step < cfg["train"]["max_steps"]:
        for x in dl:
            x = x.to(device, non_blocking=True)
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
                print(json.dumps({"step": step, "loss": float(loss.detach()), "min": (time.time() - t0) / 60}),
                      flush=True)
            if args.save_every > 0 and step % args.save_every == 0:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "cfg": cfg}, ckpt_last)
                _sync_file(ckpt_last, sync_dir)
            if step >= cfg["train"]["max_steps"]:
                break
    torch.save({"model": model.state_dict(), "cfg": cfg, "steps": step}, out / "model.pt")
    _sync_file(out / "model.pt", sync_dir)

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
    for fn in ("metrics.json", "per_image.csv", "samples.png", "first4.npy"):
        _sync_file(out / fn, sync_dir)
    print(json.dumps(res["metrics"], indent=1), flush=True)


if __name__ == "__main__":
    main()
