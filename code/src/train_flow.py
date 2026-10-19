"""训练无条件流匹配先验。

用法：
    python src/train_flow.py --config configs/prior_256.yaml
    python src/train_flow.py --config configs/smoke_phantom.yaml       # CPU/任意卡冒烟，几十秒

断点续训：--resume checkpoints/prior_256/last.pt
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import SliceDataset, SyntheticPhantomDataset
from src.flow import fm_loss, sample_uncond
from src.models.unet import UNet, count_params
from src.utils import save_grid, set_seed


def build_dataset(cfg):
    d = cfg["data"]
    if d["type"] == "phantom":
        return SyntheticPhantomDataset(n=d.get("n", 512), size=cfg["model"]["image_size"])
    return SliceDataset(d["processed_dir"], split=d.get("split", "train"), augment=True,
                        max_items=d.get("max_items"))


def ema_update(ema, model, decay):
    with torch.no_grad():
        pe = list(ema.parameters()); pm = [p.detach() for p in model.parameters()]
        torch._foreach_mul_(pe, decay)
        torch._foreach_add_(pe, pm, alpha=1 - decay)


def lr_at(step: int, cfg_train: dict) -> float:
    """线性 warmup 后余弦衰减到 lr_min（未设 lr_min 时保持常数，兼容旧配置）。"""
    base, warmup = cfg_train["lr"], cfg_train.get("warmup", 1000)
    if step < warmup:
        return base * (step + 1) / warmup
    lr_min = cfg_train.get("lr_min")
    if lr_min is None:
        return base
    prog = min(1.0, (step - warmup) / max(1, cfg_train["max_steps"] - warmup))
    return lr_min + 0.5 * (base - lr_min) * (1 + math.cos(math.pi * prog))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--max-steps", type=int, default=None, help="覆盖配置，调试用")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.max_steps:
        cfg["train"]["max_steps"] = args.max_steps

    set_seed(cfg.get("seed", 0))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(cfg["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "samples").mkdir(exist_ok=True)
    yaml.safe_dump(cfg, open(out_dir / "config.yaml", "w"))

    ds = build_dataset(cfg)
    dl = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True, drop_last=True,
                    num_workers=cfg["train"].get("num_workers", 4), pin_memory=device == "cuda")
    model = UNet(**cfg["model"]).to(device).to(memory_format=torch.channels_last)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    print(f"params: {count_params(model):.2f}M  device={device}  dataset={len(ds)}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"].get("wd", 0.0),
                            betas=(0.9, 0.99), fused=device == "cuda")
    # bf16 指数位与 fp32 相同，不需要 GradScaler
    use_amp = device == "cuda" and cfg["train"].get("amp", True)
    # 只编译训练用的前向；EMA 与保存都用未编译的 model，state_dict 不带 _orig_mod 前缀。
    # 采样（含 dps/pigdm 的 autograd.grad）一律走 eager 模型。
    fwd = torch.compile(model) if cfg["train"].get("compile", False) and device == "cuda" else model
    step, log = 0, []
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"])
        step = ck["step"]; print("resumed at step", step)

    max_steps = cfg["train"]["max_steps"]
    max_hours = cfg["train"].get("max_hours")  # 墙钟上限（小时）；到点后按当前步保存 ema_final 并结束
    accum = cfg["train"].get("grad_accum", 1)
    assert cfg["train"]["batch_size"] % accum == 0, "batch_size 须能被 grad_accum 整除"
    t0 = time.time()
    model.train()
    while step < max_steps:
        for x in dl:
            x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            lr = lr_at(step, cfg["train"])
            for g in opt.param_groups:
                g["lr"] = lr
            # grad_accum > 1：把 batch 切成等大微批逐个反传，梯度与整批一次反传等价（loss 按微批数平均），
            # 用于 24G 卡放不下 prior_256 的 batch 32
            opt.zero_grad(set_to_none=True)
            loss = 0.0
            for xc in x.chunk(accum):
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    lc = fm_loss(fwd, xc, cfg["train"].get("t_mode", "logit-normal")) / accum
                lc.backward()
                loss = loss + lc.detach()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema_update(ema, model, cfg["train"].get("ema", 0.9995))
            step += 1
            if step % cfg["train"].get("log_every", 50) == 0:
                rec = {"step": step, "loss": float(loss.detach()), "lr": lr, "min": (time.time() - t0) / 60}
                log.append(rec); print(json.dumps(rec))
                json.dump(log, open(out_dir / "train_log.json", "w"))
            if step % cfg["train"].get("sample_every", 2000) == 0 or step == max_steps:
                ema.eval()
                xs = sample_uncond(ema, (8, 1, cfg["model"]["image_size"], cfg["model"]["image_size"]),
                                   nfe=cfg["train"].get("sample_nfe", 50), device=device)
                save_grid(xs, out_dir / "samples" / f"step{step:07d}.png", nrow=4)
                torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                            "step": step, "cfg": cfg}, out_dir / "last.pt")
            if max_hours is not None and (time.time() - t0) / 3600 >= max_hours:
                print(f"[STOP] 到达墙钟上限 {max_hours} h，step={step}")
                max_steps = step
            if step >= max_steps:
                break
    torch.save({"ema": ema.state_dict(), "cfg": cfg, "step": step}, out_dir / "ema_final.pt")
    print("done ->", out_dir)


if __name__ == "__main__":
    main()
