"""训练角度序列补全器 C_φ（Phase 1b，设计见 docs/02_phase1b_angle_ssm_design.md §4）。

先验冻结；每个 batch：
    1. 随机采样一种采样模式（稀疏视角：180 的因数个视角 + 随机相位；限角：随机长度与起点，环形）。
    2. 真实切片 x → 全角度正弦图 s_true = A_full x；实测 y_full = s_true 的已测行（未测行置 0）。
    3. t ~ U(t_min, 1)，x_t = (1−t)ε + t x，x̂₁ = x_t + (1−t) v_θ(x_t, t)；以概率 p_dc 再对 x̂₁ 做一次只在已测行上的
       近端 DC，模拟采样时补全器看到的"已被数据一致性拉回"的估计。s_pred = A_full x̂₁。
    4. (ŷ, log σ²) = C_φ(s_pred, y_full, mask, t)；损失只在未测行上计算：
       MSE + 异方差 NLL（mse_θ/σ²_θ + log σ²_θ）+ β·HLCC 残差。误差按补全器内部同样的逐样本尺度归一化。

用法：python src/train_completer.py --config configs/completer_ssm.yaml [--max-steps N]
输出：<out_dir>/completer.pt（{"model", "cfg", "step"}）、train_log.json、val_log.json
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import SliceDataset, SyntheticPhantomDataset
from src.models.angle_ssm import AngleCompleter, count_params
from src.physics.hlcc import hlcc_residual
from src.physics.radon import CTOperator, interp_missing_rows, weighted_cg_dc
from src.train_flow import lr_at
from src.utils import load_ema, set_seed

N_FULL = 180
SPARSE_VIEWS = [18, 20, 30, 36, 45, 60, 90]  # 180 的因数：已测角度恰好落在全网格上


def sample_mask(rng: random.Random, p_limited: float, la_range=(60, 150)) -> tuple[torch.Tensor, str]:
    """返回 ([180] bool 掩码, 描述)。限角时缺口可以跨过 π（环形），对应 s(θ+π, r) = s(θ, −r)。"""
    m = torch.zeros(N_FULL, dtype=torch.bool)
    if rng.random() < p_limited:
        L = rng.randint(*la_range); start = rng.randrange(N_FULL)
        m[[(start + i) % N_FULL for i in range(L)]] = True
        return m, f"la{L}"
    k = rng.choice(SPARSE_VIEWS); step = N_FULL // k; phase = rng.randrange(step)
    m[phase::step] = True
    return m, f"sv{k}"


def fixed_mask(name: str) -> torch.Tensor:
    m = torch.zeros(N_FULL, dtype=torch.bool)
    if name.startswith("sv"):
        m[:: N_FULL // int(name[2:])] = True
    else:
        m[: int(name[2:])] = True
    return m


@torch.no_grad()
def make_inputs(prior, op, x, mask, t, p_dc: float, dc_lam: float):
    """构造一个 batch 的 (s_true, y_full, s_pred)。所有张量 [B,1,180,D]。"""
    s_true = op(x)
    y_full = s_true * mask.view(1, 1, -1, 1)
    eps = torch.randn_like(x)
    tt = t.view(-1, 1, 1, 1)
    xt = (1 - tt) * eps + tt * x
    x1 = (xt + (1 - tt) * prior(xt, t)).clamp(-1, 1)
    if p_dc > 0:
        sel = torch.rand(x.shape[0], device=x.device) < p_dc
        if sel.any():
            w = mask.to(x.dtype).view(1, -1).expand(int(sel.sum()), -1)
            x1[sel] = weighted_cg_dc(op, x1[sel], y_full[sel], w, lam=dc_lam, n_iter=5).clamp(-1, 1)
    return s_true, y_full, op(x1)


def loss_fn(y_hat, logvar, s_true, y_full, mask, beta_hlcc: float):
    m = mask.to(y_hat.dtype)
    # 与补全器内部一致的逐样本尺度：已测行的均方根
    scale = ((y_full[:, 0] ** 2 * m[None, :, None]).sum((1, 2)) / (m.sum() * y_hat.shape[-1])).clamp(min=1e-6).sqrt()
    err = ((y_hat - s_true)[:, 0] / scale.view(-1, 1, 1)) ** 2  # [B, A, D]
    row_mse = err.mean(-1)  # [B, A]
    un = ~mask
    mse = row_mse[:, un].mean()
    nll = (row_mse[:, un] / logvar[:, un].exp() + logvar[:, un]).mean()
    hl = hlcc_residual(y_hat).mean() if beta_hlcc > 0 else y_hat.new_zeros(())
    return mse + nll + beta_hlcc * hl, {"mse": float(mse), "nll": float(nll), "hlcc": float(hl)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.max_steps:
        cfg["train"]["max_steps"] = args.max_steps
    tc = cfg["train"]
    set_seed(cfg.get("seed", 0)); rng = random.Random(cfg.get("seed", 0))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(cfg["out_dir"]); out.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(cfg, open(out / "config.yaml", "w"))

    prior, pcfg = load_ema(cfg["prior_ckpt"], device)
    n = pcfg["model"]["image_size"]
    op = CTOperator(n, n_angles=N_FULL, device=device)
    d = cfg["data"]
    if d["type"] == "phantom":
        tr = SyntheticPhantomDataset(n=d.get("n", 256), size=n); va = SyntheticPhantomDataset(n=8, size=n, seed=7)
    else:
        tr = SliceDataset(d["processed_dir"], "train", augment=True)
        va = SliceDataset(d["processed_dir"], "val", augment=False, max_items=d.get("n_val", 32), subsample="even")
    dl = DataLoader(tr, batch_size=tc["batch_size"], shuffle=True, drop_last=True,
                    num_workers=tc.get("num_workers", 4), pin_memory=device == "cuda")
    model = AngleCompleter(n_det=n, **cfg["model"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc.get("wd", 0.0))
    print(f"completer {cfg['model'].get('mixer', 'ssm')}: {count_params(model):.2f}M params, device={device}")

    step, t0, log, vlog = 0, time.time(), [], []
    max_steps = tc["max_steps"]
    model.train()
    while step < max_steps:
        for x in dl:
            x = x.to(device, non_blocking=True)
            mask, desc = sample_mask(rng, tc.get("p_limited", 0.4))
            mask = mask.to(device)
            t = torch.rand(x.shape[0], device=device) * (1 - tc.get("t_min", 0.05)) + tc.get("t_min", 0.05)
            s_true, y_full, s_pred = make_inputs(prior, op, x, mask, t, tc.get("p_dc", 0.5), tc.get("dc_lam", 0.3))
            for g in opt.param_groups:
                g["lr"] = lr_at(step, tc)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda" and tc.get("amp", True)):
                y_hat, logvar = model(s_pred, y_full, mask, t)
            loss, parts = loss_fn(y_hat.float(), logvar.float(), s_true, y_full, mask, tc.get("beta_hlcc", 0.0))
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            step += 1
            if step % tc.get("log_every", 100) == 0:
                rec = {"step": step, "loss": float(loss), **parts, "mask": desc, "min": (time.time() - t0) / 60}
                log.append(rec); print(json.dumps(rec), flush=True)
                json.dump(log, open(out / "train_log.json", "w"))
            if step % tc.get("val_every", 1000) == 0 or step == max_steps:
                vlog.append({"step": step, **validate(model, prior, op, va, device, tc)})
                print("[VAL]", json.dumps(vlog[-1]), flush=True)
                json.dump(vlog, open(out / "val_log.json", "w"))
                torch.save({"model": model.state_dict(), "cfg": cfg, "step": step}, out / "completer.pt")
                model.train()
            if step >= max_steps:
                break
    print("done ->", out)


@torch.no_grad()
def validate(model, prior, op, va, device, tc):
    """固定掩码（sv60 / sv30 / la120 / la90）、固定 t=0.8 下，未测行的相对误差：补全器 vs 环形插值 vs 先验预测。"""
    model.eval()
    x = torch.stack([va[i] for i in range(len(va))]).to(device)
    res = {}
    for name in ("sv60", "sv30", "la120", "la90"):
        mask = fixed_mask(name).to(device)
        torch.manual_seed(0)
        t = torch.full((x.shape[0],), 0.8, device=device)
        s_true, y_full, s_pred = make_inputs(prior, op, x, mask, t, 1.0, tc.get("dc_lam", 0.3))
        y_hat, _ = model(s_pred, y_full, mask, t)
        un = ~mask
        den = s_true[..., un, :].norm()
        res[name] = {"completer": float((y_hat - s_true)[..., un, :].norm() / den),
                     "interp": float((interp_missing_rows(y_full, mask) - s_true)[..., un, :].norm() / den),
                     "prior_pred": float((s_pred - s_true)[..., un, :].norm() / den)}
    return res


if __name__ == "__main__":
    main()
