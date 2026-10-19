#!/usr/bin/env bash
# Phase 1 一键流程（AutoDL）。按阶段注释掉已完成的步骤重复运行即可。
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD:${PYTHONPATH:-}
DATA=${MFP_DATA_ROOT:-/root/autodl-tmp/data}

echo "== W1-a 物理算子自检"; python tests/test_radon.py
echo "== W1-b 冒烟（相位体）"; python src/train_flow.py --config configs/smoke_phantom.yaml
python src/sample_posterior.py --ckpt checkpoints/smoke_phantom/ema_final.pt --config configs/eval_smoke.yaml --guidance dps --out results/smoke_dps

echo "== W1-c 数据预处理"; python src/data/prepare_slices.py --raw $DATA/totalseg/raw/extracted --out $DATA/totalseg/processed
echo "== W1-d FBP 基线"
for c in sparse30 sparse60 sparse90 lowdose; do python src/baselines/fbp_eval.py --config configs/eval_$c.yaml --out results/w1_fbp_$c; done

echo "== W2-a 先验训练（后台，用 nohup / tmux）"
echo "nohup python src/train_flow.py --config configs/prior_256.yaml > checkpoints/prior_256.log 2>&1 &"
echo "== W2-b 后验采样（先验训练完成后）"
echo "for g in dps pigdm dc; do python src/sample_posterior.py --ckpt checkpoints/prior_256/ema_final.pt --config configs/eval_sparse60.yaml --guidance \$g --out results/w2_\${g}_sv60; done"
echo "== W2-c 监督基线"; echo "python src/baselines/train_fbpconvnet.py --config configs/fbpconvnet_sv60.yaml"
