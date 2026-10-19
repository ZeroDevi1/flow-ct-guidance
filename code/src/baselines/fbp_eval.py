"""B0：FBP 基线评测。用法：python src/baselines/fbp_eval.py --config configs/eval_sparse60.yaml --out results/w1_fbp_sv60"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from src.data.dataset import SliceDataset, SyntheticPhantomDataset
from src.eval.metrics import data_consistency, lpips_dist, psnr, ssim, summarize
from src.sample_posterior import build_operator, make_measurement
from src.utils import save_first, save_grid, save_per_image, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--n-test", type=int, default=None)
    ap.add_argument("--filter", default="ram-lak", choices=["ram-lak", "shepp-logan", "cosine"])
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    set_seed(cfg.get("seed", 0))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n = args.size
    n_test = args.n_test or cfg["data"].get("n_test", 200)
    op = build_operator(cfg, n, device)
    if cfg["data"]["type"] == "phantom":
        ds = SyntheticPhantomDataset(n=n_test, size=n, seed=123)
    else:
        ds = SliceDataset(cfg["data"]["processed_dir"], split=cfg["data"].get("split", "test"), augment=False,
                          max_items=n_test)
    agg = {"psnr": [], "ssim": [], "lpips": [], "dc": []}
    first = None
    for i0 in range(0, len(ds), 16):
        idx = list(range(i0, min(i0 + 16, len(ds))))
        x = torch.stack([ds[j] for j in idx]).to(device)
        y = make_measurement(op, x, cfg, indices=idx)
        xr = op.fbp(y, kind=args.filter)
        agg["psnr"] += psnr(xr, x, op.mask).tolist(); agg["ssim"] += ssim(xr, x, op.mask).tolist()
        agg["lpips"] += lpips_dist(xr, x).tolist()
        agg["dc"] += data_consistency(op, xr, y).tolist()
        if first is None:
            first = (x[:4], xr[:4])
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    res = {"method": f"fbp-{args.filter}", "physics": cfg["physics"], "metrics": summarize(agg), "n_test": len(ds)}
    json.dump(res, open(out / "metrics.json", "w"), indent=2)
    save_grid(torch.cat(first), out / "samples.png", nrow=4)
    save_first(out, list(first)); save_per_image(out, agg)
    print(json.dumps(res["metrics"], indent=1))


if __name__ == "__main__":
    main()
