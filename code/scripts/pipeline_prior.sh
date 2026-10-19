#!/usr/bin/env bash
# 数据就绪 → 预处理 → 先验训练 的串行流水线（远端 tmux 中运行，在 01_diffusion_inverse 目录下执行）。
# 用法：bash scripts/pipeline_prior.sh <下载任务的日志路径>
#   等待下载日志出现 [DONE]（下载 runner 退出却没有 [DONE] 时终止），
#   预处理后核对 meta.csv 官方划分是否全部匹配；不全匹配则停止，不启动训练。
set -euo pipefail
DL_LOG="$1"
DATA=${MFP_DATA_ROOT:-/root/autodl-tmp/data}/totalseg
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1

echo "[START] 等待下载完成：$DL_LOG"
# 以 autodl-remote 的 job 状态文件（与日志同名、位于 ../jobs/<name>.status）判断下载任务是否结束：
# "runner exited" 只打印在 tmux 面板里，不进日志文件；aria2 对可重试错误也会打印 [ERROR]，都不能用来判失败。
STATUS="$(dirname "$(dirname "$DL_LOG")")/jobs/$(basename "$DL_LOG" .log).status"
until grep -q "^status=exited" "$STATUS" 2>/dev/null; do sleep 30; done
if ! grep -q "^exit_code=0$" "$STATUS" || ! grep -q "\[DONE\]" "$DL_LOG"; then
  echo "[ERROR] 下载任务失败（$(grep exit_code "$STATUS")），见 $DL_LOG"; exit 1
fi

echo "[PROGRESS] 预处理"
python -u src/data/prepare_slices.py --raw "$DATA/raw/extracted" --out "$DATA/processed" --max-per-case 40 --workers 16 \
  | tee "$DATA/processed_prepare.log"
n_ct=$(find "$DATA/raw/extracted" -name "*.nii.gz" | wc -l)
if ! grep -q "meta.csv 划分匹配 ${n_ct}/${n_ct} 例" "$DATA/processed_prepare.log"; then
  echo "[ERROR] 官方划分未全部匹配（期望 ${n_ct}/${n_ct}），停止，不启动训练"; exit 1
fi
cat "$DATA/processed/stats.json"

echo "[PROGRESS] 先验训练"
python -u src/train_flow.py --config configs/prior_256.yaml
echo "[DONE]"
