"""在 Colab 纯 CPU（推荐 High-RAM）实例上整理 TotalSegmentator 数据与 Phase 1 权重到 Google Drive。

用法（在 Colab 会话中，仓库 code/ 目录下）：
    python scripts/colab_prep_data.py --drive-root /content/drive/MyDrive/flow-ct-guidance
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

RELEASE_BASE = "https://github.com/ZeroDevi1/flow-ct-guidance/releases/download/v1.0-phase1"
WEIGHT_FILES = [
    "SHA256SUMS.txt",
    "prior_256_ema_final.pt",
    "fbpconvnet_sv30.pt",
    "fbpconvnet_sv60.pt",
    "fbpconvnet_sv90.pt",
    "fbpconvnet_ld1e5.pt",
]
HF_REPO = "YongchengYAO/TotalSegmentator-CT-Lite"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_url(url: str, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        print(f"[SKIP] {dst} ({dst.stat().st_size / 1e6:.1f} MB)", flush=True)
        return
    print(f"[GET] {url} -> {dst}", flush=True)
    urllib.request.urlretrieve(url, dst)


def sync_to_drive(work_dir: Path, drive_root: Path):
    if not drive_root.parent.exists():
        raise SystemExit(f"未检测到 Google Drive 挂载路径 {drive_root.parent}，请先执行 colab drivemount。")
    drive_root.mkdir(parents=True, exist_ok=True)
    wdir = work_dir / "weights"
    if (wdir / "prior_256_ema_final.pt").exists():
        dst_ckpt = drive_root / "checkpoints"
        (dst_ckpt / "prior_256").mkdir(parents=True, exist_ok=True)
        shutil.copy2(wdir / "prior_256_ema_final.pt", dst_ckpt / "prior_256" / "ema_final.pt")
        for name in WEIGHT_FILES[2:]:
            if (wdir / name).exists():
                shutil.copy2(wdir / name, dst_ckpt / name)
        if (wdir / "SHA256SUMS.txt").exists():
            shutil.copy2(wdir / "SHA256SUMS.txt", dst_ckpt / "SHA256SUMS.txt")
        print(f"[SYNC] 权重已写入 {dst_ckpt}", flush=True)
    proc_dir = work_dir / "processed"
    if (proc_dir / "slices_train.npy").exists():
        dst_proc = drive_root / "processed"
        dst_proc.mkdir(parents=True, exist_ok=True)
        for p in sorted(proc_dir.iterdir()):
            if p.is_file():
                dst = dst_proc / p.name
                if dst.exists() and dst.stat().st_size == p.stat().st_size:
                    print(f"[SKIP-SYNC] {p.name} 大小一致", flush=True)
                    continue
                print(f"[SYNC] {p.name} ({p.stat().st_size / 1e6:.1f} MB) -> {dst_proc}", flush=True)
                shutil.copy2(p, dst)
        print(f"[DONE] 数据已同步至 {dst_proc}", flush=True)


def prepare_weights(work_dir: Path):
    wdir = work_dir / "weights"
    wdir.mkdir(parents=True, exist_ok=True)
    for name in WEIGHT_FILES:
        download_url(f"{RELEASE_BASE}/{name}", wdir / name)

    # 校验 SHA-256
    sums = {}
    for line in (wdir / "SHA256SUMS.txt").read_text().splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            sums[Path(parts[-1]).name] = parts[0]
    for name in WEIGHT_FILES[1:]:
        p = wdir / name
        got = sha256_file(p)
        exp = sums.get(name)
        if exp and got != exp:
            raise RuntimeError(f"SHA-256 校验失败: {name} got={got} expected={exp}")
        print(f"[OK] {name} sha256={got[:12]}...", flush=True)


def prepare_dataset(work_dir: Path, workers: int, force: bool):
    proc_dir = work_dir / "processed"
    if not force and (proc_dir / "slices_train.npy").exists() and (proc_dir / "slices_test200.npy").exists():
        print(f"[SKIP] 本地已有预处理数据: {proc_dir}", flush=True)
        return

    from huggingface_hub import hf_hub_download

    raw_dir = work_dir / "raw"
    ext_dir = raw_dir / "extracted"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"[HF] 下载 {HF_REPO} (Images.zip, meta.csv)...", flush=True)
    meta_path = Path(hf_hub_download(repo_id=HF_REPO, repo_type="dataset", filename="meta.csv",
                                     local_dir=str(raw_dir)))
    zip_path = Path(hf_hub_download(repo_id=HF_REPO, repo_type="dataset", filename="Images.zip",
                                    local_dir=str(raw_dir)))
    print(f"[HF] 下载完成: zip={zip_path.stat().st_size / 1e9:.2f} GB", flush=True)

    if not ext_dir.exists() or not any(ext_dir.iterdir()):
        ext_dir.mkdir(parents=True, exist_ok=True)
        print(f"[UNZIP] 解压 {zip_path} -> {ext_dir}...", flush=True)
        subprocess.run(["unzip", "-q", str(zip_path), "-d", str(ext_dir)], check=True)
    shutil.copy2(meta_path, ext_dir / "meta.csv")

    print(f"[PREP] 运行 prepare_slices.py (workers={workers})...", flush=True)
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])}
    subprocess.run(
        [sys.executable, "-u", "src/data/prepare_slices.py",
         "--raw", str(ext_dir), "--out", str(proc_dir),
         "--max-per-case", "40", "--workers", str(workers)],
        check=True, env=env,
    )
    stats = json.loads((proc_dir / "stats.json").read_text())
    print("[STATS]", json.dumps(stats, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive-root", default="/content/drive/MyDrive/flow-ct-guidance")
    ap.add_argument("--work-dir", default="/content/work_prep")
    ap.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--weights-only", action="store_true")
    ap.add_argument("--sync-only", action="store_true", help="仅将 work-dir 中的产物同步到 Drive")
    ap.add_argument("--force", action="store_true", help="即使本地已有 processed 也重新处理")
    args = ap.parse_args()

    drive_root = Path(args.drive_root)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    if args.sync_only:
        sync_to_drive(work_dir, drive_root)
        return

    prepare_weights(work_dir)
    if not args.weights_only:
        prepare_dataset(work_dir, args.workers, force=args.force)
    if drive_root.parent.exists():
        sync_to_drive(work_dir, drive_root)
    else:
        print(f"[INFO] 尚未挂载 {drive_root.parent}；挂载后运行 --sync-only 即可复制到 Drive。", flush=True)


if __name__ == "__main__":
    main()
