"""Track 2-1：训练 2D 测量条件化流（冻结 19.8M 流先验 + 零卷积 ControlNet 分支）。

特性：
1. 共享单组 180 角预计算旋转网格（_grid/_back_grid），零显存冗余地即时生成
   la120 / la90 / 随机楔限角 / 稀疏视角 / 低剂量 Poisson 多几何训练样本。
2. 支持 --profile：上卡前 15 秒逐模块剖析（数据/物理/前向/反传/显存/预估 10k 步耗时与 CU）。
3. 支持 --resume 与 --sync-dir（断点、日志、中间权重自动同步 Google Drive）。
4. 定期在 val 探针（la120 / la90，纯条件 ODE 与条件+Prox-DC）上记录 PSNR 曲线。

用法：
    python src/train_control_flow.py --config configs/control_flow_t2_1.yaml --profile
    python src/train_control_flow.py --config configs/control_flow_t2_1.yaml \
        --prior checkpoints/prior_256/ema_final.pt --data-dir /content/data/processed \
        --sync-dir /content/drive/MyDrive/flow-ct-guidance/track2_1/ckpt --resume
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import shutil
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import SliceDataset, SyntheticPhantomDataset
from src.eval.metrics import psnr
from src.flow import sample_t
from src.models.control_flow import (
    BoundControlFlow,
    ControlFlowUNet,
    angle_coverage_vector,
    build_cond_context,
    fourier_wedge_mask,
)
from src.models.unet import count_params
from src.physics.radon import CTOperator, circle_mask, fbp, poisson_ray_variance, radon
from src.sample_posterior import posterior_sample
from src.train_flow import lr_at
from src.utils import load_ema, save_grid, set_seed


class MultiGeometrySampler:
    """复用单组 180 角网格与精确标定 fbp_scale 的多几何在线 CT 物理采样器。"""

    def __init__(self, n: int, mix_cfg: dict, device: str):
        self.n = n
        self.device = device
        self.mix_cfg = mix_cfg
        self.op_full = CTOperator(n, n_angles=180, device=device)
        self.inner_mask = circle_mask(n, device=device, dtype=torch.float32, radius=0.8).bool()
        self.target_inner = float((self.op_full.mask / 2)[self.inner_mask].mean().item())

        # 预构建并缓存典型固定几何的 CTOperator + Fourier 掩码（与测试配置 100% 比特一致）
        self.ops = {
            "la120": CTOperator(n, n_angles=120, angle_range=2.0 * math.pi / 3.0, device=device),
            "la90": CTOperator(n, n_angles=90, angle_range=0.5 * math.pi, device=device),
            "sv30": CTOperator(n, n_angles=30, angle_range=math.pi, device=device),
            "sv60": CTOperator(n, n_angles=60, angle_range=math.pi, device=device),
            "sv90": CTOperator(n, n_angles=90, angle_range=math.pi, device=device),
            "full180": self.op_full,
        }
        self.fmasks = {
            k: fourier_wedge_mask(n, op.angles, device=device, dtype=torch.float32)
            for k, op in self.ops.items()
        }
        # 预缓存不同宽度的随机限角 fbp_scale（旋转对称，与起始角几乎无关，仅需按宽度标定一次）
        self.la_min = int(mix_cfg.get("la_rand_min_deg", 80))
        self.la_max = int(mix_cfg.get("la_rand_max_deg", 150))
        self._width_fbp_scale: dict[int, float] = {}
        with torch.no_grad():
            for w in range(self.la_min, self.la_max + 1, 5):
                idx = torch.arange(0, w, device=device)
                off_sub = self.op_full.offset[:, :, idx]
                ang_sub = self.op_full.angles[idx]
                rec = fbp(off_sub, ang_sub, back_grid=self.op_full._back_grid[idx])
                self._width_fbp_scale[w] = self.target_inner / float(rec[self.inner_mask].mean().clamp(min=1e-6).item())

        self.keys = ["la120", "la90", "la_rand", "sv", "ld"]
        self.probs = [float(mix_cfg.get(k, 0.0)) for k in self.keys]
        s = sum(self.probs)
        self.probs = [p / s for p in self.probs]
        self.ld_i0_choices = [float(v) for v in mix_cfg.get("ld_i0_choices", [1e4, 3e4, 1e5, 3e5])]
        self.mu_scale = float(mix_cfg.get("mu_scale", 0.06))

    @torch.no_grad()
    def sample_batch_cond(self, x: torch.Tensor, rng: random.Random, torch_gen: torch.Generator) -> tuple[dict, str]:
        mode = rng.choices(self.keys, weights=self.probs, k=1)[0]
        if mode in ("la120", "la90"):
            op = self.ops[mode]
            y = op(x)
            return build_cond_context(op, y, physics_cfg=None, freq_mask=self.fmasks[mode]), mode

        if mode == "sv":
            sub = rng.choice(["sv30", "sv60", "sv90"])
            op = self.ops[sub]
            y = op(x)
            return build_cond_context(op, y, physics_cfg=None, freq_mask=self.fmasks[sub]), sub

        if mode == "ld":
            op = self.ops["full180"]
            i0 = rng.choice(self.ld_i0_choices)
            y_clean = op(x)
            y = op.add_noise(y_clean, i0=i0, sigma=0.0, mu_scale=self.mu_scale, generator=torch_gen)
            phys = {"i0": i0, "sigma": 0.0, "mu_scale": self.mu_scale}
            return build_cond_context(op, y, physics_cfg=phys, freq_mask=self.fmasks["full180"]), f"ld_{int(i0)}"

        # mode == "la_rand": 随机宽度（5° 步长）+ 随机起始角（1° 步长），复用 op_full 预缓存网格
        w = rng.randrange(self.la_min, self.la_max + 1, 5)
        start = rng.randrange(0, 180)
        idx = (torch.arange(start, start + w, device=self.device) % 180).long()
        angles = self.op_full.angles[idx]
        u = self.op_full.x_to_u(x)
        y = radon(u, angles, grid=self.op_full._grid[idx])
        u_fbp = fbp(y, angles, back_grid=self.op_full._back_grid[idx]) * self._width_fbp_scale[w]
        x_fbp = self.op_full.u_to_x(u_fbp).clamp(-1.0, 1.0)
        fmask = fourier_wedge_mask(self.n, angles, device=self.device, dtype=torch.float32)
        x_fbp_meas = torch.fft.irfft2(torch.fft.rfft2(x_fbp.float()) * fmask, s=x_fbp.shape[-2:])
        gvec = angle_coverage_vector(angles, n_bins=180, noise_level=0.0, device=self.device, dtype=torch.float32)
        ctx = {
            "x_fbp": x_fbp.float(),
            "x_fbp_meas": x_fbp_meas,
            "freq_mask": fmask,
            "geom_vec": gvec[None].expand(x.shape[0], -1).contiguous(),
        }
        return ctx, f"la_rand_{w}@{start}"


def _sync_file(src: Path, sync_dir: Path | None):
    if sync_dir is None or not src.exists():
        return
    sync_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, sync_dir / src.name)


def _ema_update_trainable(ema_model: ControlFlowUNet, model: ControlFlowUNet, decay: float):
    with torch.no_grad():
        pe = list(ema_model.trainable_parameters())
        pm = [p.detach() for p in model.trainable_parameters()]
        torch._foreach_mul_(pe, decay)
        torch._foreach_add_(pe, pm, alpha=1.0 - decay)


@torch.no_grad()
def run_val_probe(
    ema_model: ControlFlowUNet,
    val_x: torch.Tensor,
    geo: MultiGeometrySampler,
    nfe: int = 25,
    out_dir: Path | None = None,
    step: int = 0,
) -> dict:
    """快速验证探针：在 la120 与 la90 上测纯条件 ODE（none）与条件+Prox-DC（dc, lam=0.3）的 val PSNR。"""
    ema_model.eval()
    res = {}
    for setting in ("la120", "la90"):
        op = geo.ops[setting]
        y = op(val_x)
        x_fbp = op.fbp(y)
        ctx = build_cond_context(op, y, physics_cfg=None, x_fbp=x_fbp, freq_mask=geo.fmasks[setting])
        bound = BoundControlFlow(ema_model, ctx, cond_scale=1.0)

        p_fbp = float(psnr(x_fbp, val_x, op.mask).mean().item())
        g1 = torch.Generator(device=val_x.device).manual_seed(0)
        x_none = posterior_sample(bound, op, y, val_x.shape, guidance="none", nfe=nfe, generator=g1)
        p_none = float(psnr(x_none, val_x, op.mask).mean().item())

        g2 = torch.Generator(device=val_x.device).manual_seed(0)
        x_dc = posterior_sample(bound, op, y, val_x.shape, guidance="dc", nfe=nfe, dc_lam=0.3, dc_iters=5, generator=g2)
        p_dc = float(psnr(x_dc, val_x, op.mask).mean().item())

        res[setting] = {"fbp": round(p_fbp, 3), "cond_none": round(p_none, 3), "cond_dc0.3": round(p_dc, 3)}
        if out_dir is not None:
            k = min(4, val_x.shape[0])
            save_grid(
                torch.cat([val_x[:k], x_fbp[:k], x_none[:k], x_dc[:k]]),
                out_dir / "samples" / f"val_{setting}_step{step:06d}.png",
                nrow=k,
            )
    return res


def run_profile(model, opt, dl, geo, device, use_amp: bool, t_mode: str):
    """15 秒逐模块剖析（铁律 #3）：测数据加载、物理条件构造、前向、反传+优化器与显存峰值。"""
    rng = random.Random(0)
    tgen = torch.Generator(device=device).manual_seed(0)
    it = iter(dl)
    model.train()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def _next():
        nonlocal it
        try:
            return next(it)
        except StopIteration:
            it = iter(dl)
            return next(it)

    # 预热 5 步
    for _ in range(5):
        x = _next().to(device, non_blocking=True)
        ctx, _ = geo.sample_batch_cond(x, rng, tgen)
        t = sample_t(x.shape[0], device, t_mode)
        eps = torch.randn_like(x)
        tt = t.view(-1, 1, 1, 1)
        xt = (1.0 - tt) * eps + tt * x
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            v = model(xt, t, ctx)
            loss = ((v.float() - (x - eps)) ** 2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
        opt.step()

    if device == "cuda":
        torch.cuda.synchronize()

    n_bench = 25
    t_data = t_phys = t_fwd = t_bwd = 0.0
    for _ in range(n_bench):
        t0 = time.perf_counter()
        x = _next().to(device, non_blocking=True)
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        ctx, _ = geo.sample_batch_cond(x, rng, tgen)
        t = sample_t(x.shape[0], device, t_mode)
        eps = torch.randn_like(x)
        tt = t.view(-1, 1, 1, 1)
        xt = (1.0 - tt) * eps + tt * x
        if device == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            v = model(xt, t, ctx)
            loss = ((v.float() - (x - eps)) ** 2).mean()
        if device == "cuda":
            torch.cuda.synchronize()
        t3 = time.perf_counter()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
        opt.step()
        if device == "cuda":
            torch.cuda.synchronize()
        t4 = time.perf_counter()
        t_data += t1 - t0
        t_phys += t2 - t1
        t_fwd += t3 - t2
        t_bwd += t4 - t3

    ms_data = 1e3 * t_data / n_bench
    ms_phys = 1e3 * t_phys / n_bench
    ms_fwd = 1e3 * t_fwd / n_bench
    ms_bwd = 1e3 * t_bwd / n_bench
    ms_tot = ms_data + ms_phys + ms_fwd + ms_bwd
    hrs_10k = (ms_tot * 10000.0) / 3.6e6
    cu_10k = hrs_10k * 6.73
    vram_gb = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
    rep = {
        "batch_size": x.shape[0],
        "data_ms": round(ms_data, 2),
        "physics_ms": round(ms_phys, 2),
        "fwd_ms": round(ms_fwd, 2),
        "bwd_opt_ms": round(ms_bwd, 2),
        "total_ms": round(ms_tot, 2),
        "est_10k_hours": round(hrs_10k, 3),
        "est_10k_cu_a100": round(cu_10k, 2),
        "peak_vram_gb": round(vram_gb, 2),
    }
    print("[PROFILE]", json.dumps(rep, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--prior", default=None, help="覆盖 prior_ckpt")
    ap.add_argument("--data-dir", default=None, help="覆盖 data.processed_dir")
    ap.add_argument("--out-dir", default=None, help="覆盖 out_dir")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--sync-dir", default=None, help="可选：将断点、日志与最终权重同步到此目录（如 Drive）")
    ap.add_argument("--profile", action="store_true", help="仅跑 15 秒逐模块剖析后退出")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.prior:
        cfg["prior_ckpt"] = args.prior
    if args.data_dir:
        cfg["data"]["processed_dir"] = args.data_dir
    if args.out_dir:
        cfg["out_dir"] = args.out_dir
    if args.max_steps:
        cfg["train"]["max_steps"] = args.max_steps
    if args.batch_size:
        cfg["train"]["batch_size"] = args.batch_size

    set_seed(cfg.get("seed", 0))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "samples").mkdir(exist_ok=True)
    sync_dir = Path(args.sync_dir) if args.sync_dir else None

    # 加载冻结 19.8M 先验并构建 ControlFlowUNet
    base, mcfg = load_ema(cfg["prior_ckpt"], device=device)
    n = int(mcfg["model"]["image_size"])
    ctrl_cfg = cfg.get("control", {"cond_ch": 4, "geom_dim": 182})
    model = ControlFlowUNet(
        base,
        cond_ch=int(ctrl_cfg.get("cond_ch", 4)),
        geom_dim=int(ctrl_cfg.get("geom_dim", 182)),
    ).to(device)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)

    n_total = count_params(model)
    n_train = sum(p.numel() for p in model.trainable_parameters()) / 1e6
    print(f"ControlFlowUNet: total={n_total:.2f}M  trainable={n_train:.2f}M  frozen_base={count_params(base):.2f}M  device={device}", flush=True)

    # 数据集
    if cfg["data"].get("type") == "phantom":
        tr_ds = SyntheticPhantomDataset(n=cfg["data"].get("n", 256), size=n)
        val_ds = SyntheticPhantomDataset(n=cfg["data"].get("val_probe_n", 8), size=n, seed=123)
    else:
        tr_ds = SliceDataset(cfg["data"]["processed_dir"], split=cfg["data"].get("split", "train"), augment=True)
        val_ds = SliceDataset(
            cfg["data"]["processed_dir"],
            split=cfg["data"].get("val_split", "val"),
            augment=False,
            max_items=cfg["data"].get("val_probe_n", 16),
            subsample="even",
        )
    nw = int(cfg["train"].get("num_workers", 6))
    dl = DataLoader(
        tr_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        drop_last=True,
        num_workers=nw,
        pin_memory=(device == "cuda"),
        persistent_workers=(nw > 0),
    )
    geo = MultiGeometrySampler(n, cfg.get("geometry_mix", {}), device=device)

    opt = torch.optim.AdamW(
        list(model.trainable_parameters()),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"].get("wd", 0.0)),
        betas=(0.9, 0.99),
        fused=(device == "cuda"),
    )
    use_amp = (device == "cuda") and bool(cfg["train"].get("amp", True))
    t_mode = cfg["train"].get("t_mode", "uniform")

    if args.profile:
        run_profile(model, opt, dl, geo, device, use_amp=use_amp, t_mode=t_mode)
        return

    yaml.safe_dump(cfg, open(out_dir / "config.yaml", "w"))
    _sync_file(out_dir / "config.yaml", sync_dir)

    step = 0
    log: list[dict] = []
    val_log: list[dict] = []
    ckpt_last = out_dir / "ckpt_last.pt"
    if args.resume and not ckpt_last.exists() and sync_dir and (sync_dir / "ckpt_last.pt").exists():
        shutil.copy2(sync_dir / "ckpt_last.pt", ckpt_last)
        print(f"[RESTORE] 从同步目录拉回 {ckpt_last}", flush=True)
    if args.resume and ckpt_last.exists():
        ck = torch.load(ckpt_last, map_location=device)
        model.load_trainable_state_dict(ck["trainable"])
        ema.load_trainable_state_dict(ck["ema_trainable"])
        opt.load_state_dict(ck["opt"])
        step = int(ck["step"])
        if (out_dir / "train_log.json").exists():
            log = json.loads((out_dir / "train_log.json").read_text())
        if (out_dir / "val_log.json").exists():
            val_log = json.loads((out_dir / "val_log.json").read_text())
        print(f"[RESUME] step={step}/{cfg['train']['max_steps']}", flush=True)

    val_x = torch.stack([val_ds[i] for i in range(len(val_ds))]).to(device)
    if step == 0:
        # 第 0 步基线探针（确认零初始化与冻结无条件先验一致）
        probe0 = run_val_probe(ema, val_x, geo, nfe=int(cfg["train"].get("val_nfe", 25)), out_dir=out_dir, step=0)
        rec0 = {"step": 0, "min": 0.0, **probe0}
        val_log.append(rec0)
        print("[VAL]", json.dumps(rec0), flush=True)
        (out_dir / "val_log.json").write_text(json.dumps(val_log, indent=2))
        _sync_file(out_dir / "val_log.json", sync_dir)

    max_steps = int(cfg["train"]["max_steps"])
    ema_target = float(cfg["train"].get("ema", 0.999))
    log_every = int(cfg["train"].get("log_every", 100))
    save_every = int(cfg["train"].get("save_every", 2000))
    val_every = int(cfg["train"].get("val_every", 2500))
    val_nfe = int(cfg["train"].get("val_nfe", 25))

    rng = random.Random(cfg.get("seed", 0) + step)
    tgen = torch.Generator(device=device).manual_seed(cfg.get("seed", 0) + step + 17)
    t0 = time.time()
    loss_acc = 0.0
    loss_cnt = 0

    def _save_ckpt(path: Path, include_opt: bool = True):
        payload = {
            "trainable": model.trainable_state_dict(),
            "ema_trainable": ema.trainable_state_dict(),
            "step": step,
            "ctrl_cfg": ctrl_cfg,
            "cfg": cfg,
        }
        if include_opt:
            payload["opt"] = opt.state_dict()
        torch.save(payload, path)
        _sync_file(path, sync_dir)

    model.train()
    while step < max_steps:
        for x in dl:
            x = x.to(device, non_blocking=True)
            lr = lr_at(step, cfg["train"])
            for g in opt.param_groups:
                g["lr"] = lr

            ctx, mode_name = geo.sample_batch_cond(x, rng, tgen)
            t = sample_t(x.shape[0], device, t_mode)
            eps = torch.randn_like(x)
            tt = t.view(-1, 1, 1, 1)
            xt = (1.0 - tt) * eps + tt * x

            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                v = model(xt, t, ctx)
                loss = ((v.float() - (x - eps)) ** 2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
            opt.step()

            step += 1
            # Ramp-up EMA：避免前几千步被初始零卷积权重拖慢
            decay = min(ema_target, (1.0 + step) / (10.0 + step))
            _ema_update_trainable(ema, model, decay)

            loss_acc += float(loss.detach().item())
            loss_cnt += 1

            if step % log_every == 0:
                rec = {
                    "step": step,
                    "loss": round(loss_acc / max(loss_cnt, 1), 5),
                    "lr": round(lr, 7),
                    "min": round((time.time() - t0) / 60.0, 2),
                    "last_mode": mode_name,
                }
                log.append(rec)
                print(json.dumps(rec), flush=True)
                (out_dir / "train_log.json").write_text(json.dumps(log, indent=2))
                loss_acc = 0.0
                loss_cnt = 0

            if save_every > 0 and step % save_every == 0:
                _save_ckpt(ckpt_last, include_opt=True)
                _sync_file(out_dir / "train_log.json", sync_dir)

            if val_every > 0 and (step % val_every == 0 or step == max_steps):
                probe = run_val_probe(ema, val_x, geo, nfe=val_nfe, out_dir=out_dir, step=step)
                vrec = {"step": step, "min": round((time.time() - t0) / 60.0, 2), **probe}
                val_log.append(vrec)
                print("[VAL]", json.dumps(vrec), flush=True)
                (out_dir / "val_log.json").write_text(json.dumps(val_log, indent=2))
                _sync_file(out_dir / "val_log.json", sync_dir)
                if step in (5000, 10000, max_steps):
                    _save_ckpt(out_dir / f"ctrl_step{step}.pt", include_opt=False)
                model.train()

            if step >= max_steps:
                break

    _save_ckpt(out_dir / "control_final.pt", include_opt=False)
    _save_ckpt(ckpt_last, include_opt=True)
    _sync_file(out_dir / "train_log.json", sync_dir)
    _sync_file(out_dir / "val_log.json", sync_dir)
    print(f"[DONE] step={step} -> {out_dir} ({(time.time() - t0) / 60.0:.2f} min)", flush=True)


if __name__ == "__main__":
    main()
