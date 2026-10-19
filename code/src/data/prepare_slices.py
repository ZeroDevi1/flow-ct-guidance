"""将 TotalSegmentator（或任意 nii.gz CT 目录）转换为 256x256 归一化轴位切片。

输出：
    <out>/slices_train.npy, slices_val.npy, slices_test.npy   float16, [N, 256, 256], 取值 [-1, 1]
    <out>/index_<split>.csv                                     每张切片来源（病人 id、切片号）
    <out>/slices_test200.npy, index_test200.csv                 固定评测集（20 个 test 病人 × 10 张）
    <out>/stats.json

用法：
    python src/data/prepare_slices.py --raw /root/autodl-tmp/data/totalseg/raw/extracted \
        --out /root/autodl-tmp/data/totalseg/processed --max-per-case 40

划分优先使用 meta.csv 的 split 列；没有则按病人随机 8:1:1。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HU_MIN, HU_MAX = -1000.0, 1000.0


def hu_to_unit(x: np.ndarray) -> np.ndarray:
    """HU -> [-1, 1]，与 physics/radon.py 和模型输入一致。"""
    x = np.clip(x, HU_MIN, HU_MAX)
    return (x - HU_MIN) / (HU_MAX - HU_MIN) * 2.0 - 1.0


def center_crop_or_pad(img: np.ndarray, size: int, fill: float) -> np.ndarray:
    h, w = img.shape
    out = np.full((size, size), fill, dtype=img.dtype)
    sh, sw = max(0, (h - size) // 2), max(0, (w - size) // 2)
    dh, dw = max(0, (size - h) // 2), max(0, (size - w) // 2)
    ch, cw = min(h, size), min(w, size)
    out[dh:dh + ch, dw:dw + cw] = img[sh:sh + ch, sw:sw + cw]
    return out


def resize2d(img: np.ndarray, size: int) -> np.ndarray:
    from scipy.ndimage import zoom
    zh, zw = size / img.shape[0], size / img.shape[1]
    return zoom(img, (zh, zw), order=1)


def load_ct(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    import nibabel as nib
    im = nib.load(str(path))
    im = nib.as_closest_canonical(im)  # RAS
    arr = np.asarray(im.dataobj, dtype=np.float32)
    sp = tuple(float(s) for s in im.header.get_zooms()[:3])
    return arr, sp


def pick_slices(vol: np.ndarray, spacing, max_per_case: int, size: int, fg_thresh: float):
    """vol 为 RAS 体数据 [X, Y, Z]；轴位切片沿 Z。返回 list[(z, slice256)]。"""
    sx, sy, _ = spacing
    z_total = vol.shape[2]
    cand = []
    for z in range(z_total):
        sl = vol[:, :, z]
        fg = (sl > -500).mean()  # 体部占比
        if fg < fg_thresh:
            continue
        cand.append(z)
    if not cand:
        return []
    # 均匀抽取
    if len(cand) > max_per_case:
        idx = np.linspace(0, len(cand) - 1, max_per_case).round().astype(int)
        cand = [cand[i] for i in idx]
    out = []
    for z in cand:
        sl = vol[:, :, z].T[::-1, :]  # 转为常见的显示方向（前在上）
        # 先按物理尺寸把像素变成各向同性，再裁剪/缩放到 size
        if abs(sx - sy) > 1e-3:
            sl = resize2d(sl, int(round(sl.shape[0] * sx / min(sx, sy))))
        # 以 1.5mm 为基准：TotalSegmentator 是 1.5mm 各向同性，直接裁剪到 size
        sl = center_crop_or_pad(sl, size, fill=HU_MIN) if max(sl.shape) >= size else resize2d(sl, size)
        if sl.shape != (size, size):
            sl = resize2d(sl, size)
        out.append((z, hu_to_unit(sl).astype(np.float16)))
    return out


def _process_case(path: Path, max_per_case: int, size: int, fg_thresh: float):
    """子进程入口：读一例 CT 并抽切片；读失败返回 None 由主进程记录跳过。"""
    try:
        vol, spacing = load_ct(path)
    except Exception:  # noqa: BLE001
        return None
    return pick_slices(vol, spacing, max_per_case, size, fg_thresh)


def read_meta_split(raw: Path) -> dict[str, str]:
    metas = list(raw.rglob("meta.csv"))
    if not metas:
        return {}
    split = {}
    # utf-8-sig：官方 meta.csv 带 BOM，按 utf-8 读会把表头变成 "﻿image_id"，导致所有病人都匹配不上
    with open(metas[0], newline="", encoding="utf-8-sig") as f:
        delim = ";" if ";" in f.readline() else ","
        f.seek(0)
        for row in csv.DictReader(f, delimiter=delim):
            pid = row.get("image_id") or row.get("id") or row.get("subject")
            sp = row.get("split")
            if pid and sp:
                split[pid.strip()] = sp.strip()
    return split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--max-per-case", type=int, default=40)
    ap.add_argument("--fg-thresh", type=float, default=0.10)
    ap.add_argument("--limit-cases", type=int, default=0, help="调试用：只处理前 N 例")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--bench-patients", type=int, default=20, help="固定评测集 test200 的病人数")
    ap.add_argument("--slices-per-patient", type=int, default=10)
    args = ap.parse_args()

    random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    cts = sorted(args.raw.rglob("ct.nii.gz"))
    if not cts:  # 允许任意 nii.gz 目录
        cts = sorted(p for p in args.raw.rglob("*.nii.gz") if "seg" not in p.name.lower())
    if args.limit_cases:
        cts = cts[: args.limit_cases]
    print(f"找到 {len(cts)} 例 CT")

    meta_split = read_meta_split(args.raw)
    buffers = {"train": [], "val": [], "test": []}
    index = {"train": [], "val": [], "test": []}
    jobs, n_fallback = [], 0
    for p in cts:
        pid = p.parent.name if p.name == "ct.nii.gz" else p.stem.replace(".nii", "")
        sp = meta_split.get(pid)
        if sp not in buffers:
            n_fallback += 1
            r = random.random()
            sp = "train" if r < 0.8 else ("val" if r < 0.9 else "test")
        jobs.append((p, pid, sp))
    # meta.csv 存在却大量匹配不上，通常是 ID 命名或编码问题；打印出来，避免官方划分被随机划分悄悄替代
    print(f"meta.csv 划分匹配 {len(jobs) - n_fallback}/{len(jobs)} 例，其余按随机 8:1:1")

    # 解压 nii.gz + 插值是 CPU 瓶颈，按病例并行；imap 保持输入顺序，输出与单进程一致
    work = partial(_process_case, max_per_case=args.max_per_case, size=args.size, fg_thresh=args.fg_thresh)
    with Pool(args.workers) as pool:
        for i, ((p, pid, sp), slices) in enumerate(zip(jobs, pool.imap(work, [j[0] for j in jobs], chunksize=4))):
            if slices is None:
                print(f"[skip] {p}")
                continue
            for z, sl in slices:
                buffers[sp].append(sl)
                index[sp].append((pid, z))
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(jobs)}  train={len(buffers['train'])} val={len(buffers['val'])} "
                      f"test={len(buffers['test'])}", flush=True)

    stats = {}
    for sp, buf in buffers.items():
        arr = np.stack(buf) if buf else np.zeros((0, args.size, args.size), np.float16)
        np.save(args.out / f"slices_{sp}.npy", arr)
        with open(args.out / f"index_{sp}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["patient_id", "z"])
            w.writerows(index[sp])
        stats[sp] = {"n_slices": int(arr.shape[0]), "n_patients": len({pid for pid, _ in index[sp]})}
        print(sp, stats[sp])

    # 固定评测集 test200：从 test 划分中随机选 bench_patients 个病人，每人沿 z 均匀取 slices_per_patient 张。
    # 所有方法共用这一份文件，避免"取前 N 张"只覆盖少数几个病人。
    if buffers["test"]:
        rng = random.Random(args.seed)
        by_pid: dict[str, list[int]] = {}
        for k, (pid, _) in enumerate(index["test"]):
            by_pid.setdefault(pid, []).append(k)
        pids = sorted(pid for pid, ks in by_pid.items() if len(ks) >= args.slices_per_patient)
        chosen = sorted(rng.sample(pids, min(args.bench_patients, len(pids))))
        sel = []
        for pid in chosen:
            ks = by_pid[pid]
            sel += [ks[j] for j in np.linspace(0, len(ks) - 1, args.slices_per_patient).round().astype(int)]
        np.save(args.out / "slices_test200.npy", np.stack([buffers["test"][k] for k in sel]))
        with open(args.out / "index_test200.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["patient_id", "z"])
            w.writerows(index["test"][k] for k in sel)
        stats["test200"] = {"n_slices": len(sel), "n_patients": len(chosen)}
        print("test200", stats["test200"])
    stats.update({"size": args.size, "hu_window": [HU_MIN, HU_MAX], "value_range": [-1, 1]})
    with open(args.out / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("完成 ->", args.out)


if __name__ == "__main__":
    main()
