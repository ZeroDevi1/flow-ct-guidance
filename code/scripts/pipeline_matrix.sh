#!/usr/bin/env bash
# 先验训练结束 → 后验采样全矩阵 → FBPConvNet → 汇总 的串行流水线（远端 tmux 中运行，在 01_diffusion_inverse 目录下执行）。
# 用法：bash scripts/pipeline_matrix.sh <先验训练任务的 status 文件>
#   等该任务 status=exited；exit_code≠0 或缺少 ema_final.pt 时终止，不进入采样。
#   run_matrix.py 对已有 metrics.json 的实验自动跳过，本脚本可中断后重跑。
set -euo pipefail
STATUS="$1"
CKPT=checkpoints/prior_256/ema_final.pt
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1

echo "[START] 等待先验训练结束：$STATUS"
until grep -q "^status=exited" "$STATUS" 2>/dev/null; do sleep 60; done
if ! grep -q "^exit_code=0$" "$STATUS" || [ ! -f "$CKPT" ]; then
  echo "[ERROR] 先验训练未正常结束（$(grep exit_code "$STATUS")），停止"; exit 1
fi

python -u scripts/run_matrix.py --ckpt "$CKPT" --stage tune select main nfe unc ld1e4 fbpconvnet
python scripts/collect_results.py
echo "[DONE]"
