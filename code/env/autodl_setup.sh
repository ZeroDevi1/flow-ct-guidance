#!/usr/bin/env bash
# =============================================================================
# AutoDL 环境初始化脚本 · MedFusionProjects
#
# 用法（在 AutoDL 实例上，以 root 身份）：
#   git clone <你的仓库> /root/MedFusionProjects   # 或 scp 上传
#   bash /root/MedFusionProjects/shared/env/autodl_setup.sh
#
# 前提：选择 AutoDL 官方镜像 PyTorch >= 2.3 / CUDA >= 12.1（镜像自带 torch，脚本不重装）。
# 数据盘：/root/autodl-tmp（实例重置后保留），所有数据与权重放这里。
# =============================================================================
set -euo pipefail

MFP_ROOT="${MFP_ROOT:-/root/MedFusionProjects}"
DATA_ROOT="${MFP_DATA_ROOT:-/root/autodl-tmp/data}"
CKPT_ROOT="/root/autodl-tmp/checkpoints"

echo "== 0. 学术加速（AutoDL 提供，访问 GitHub / HF / PyPI 国外源更快）"
if [ -f /etc/network_turbo ]; then
  # shellcheck disable=SC1091
  source /etc/network_turbo || true
fi

echo "== 1. 基础目录"
mkdir -p "$DATA_ROOT" "$CKPT_ROOT"
# 仓库内的 checkpoints/ 软链到数据盘，避免系统盘写满
for d in "$MFP_ROOT"/0*_*/checkpoints; do
  [ -d "$d" ] || continue
  if [ ! -L "$d" ]; then
    proj=$(basename "$(dirname "$d")")
    mkdir -p "$CKPT_ROOT/$proj"
    rmdir "$d" 2>/dev/null || mv "$d" "$CKPT_ROOT/${proj}_old_$(date +%s)"
    ln -s "$CKPT_ROOT/$proj" "$d"
  fi
done

echo "== 2. Python 依赖（镜像已有 torch/torchvision，这里只补医学影像与实验工具）"
pip install -q -r "$MFP_ROOT/shared/env/requirements.txt"

echo "== 3. 环境变量写入 ~/.bashrc"
grep -q "MFP_DATA_ROOT" ~/.bashrc || cat >> ~/.bashrc <<EOF

# --- MedFusionProjects ---
export MFP_ROOT="$MFP_ROOT"
export MFP_DATA_ROOT="$DATA_ROOT"
export HF_HOME="/root/autodl-tmp/hf_cache"
export HF_ENDPOINT="https://hf-mirror.com"   # 国内镜像；有学术加速时可注释掉
export PYTHONPATH="\$MFP_ROOT/01_diffusion_inverse:\$PYTHONPATH"
EOF

echo "== 4. （可选）安装 Claude Code 以便在远程机器上直接跑 claude"
# 需要 Node.js >= 18。AutoDL 镜像通常没有 node，用 nvm 安装：
if ! command -v node >/dev/null 2>&1; then
  echo "   未检测到 node，若需要远程 Claude Code，请执行："
  echo "   curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash && source ~/.bashrc && nvm install 22"
  echo "   npm install -g @anthropic-ai/claude-code"
  echo "   （国内网络请先 source /etc/network_turbo，或配置 npm 镜像）"
fi

echo "== 5. 自检"
python - <<'PY'
import torch, importlib
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
for m in ["nibabel", "monai", "skimage", "lpips", "yaml", "tqdm", "scipy"]:
    try:
        importlib.import_module(m); print("ok ", m)
    except Exception as e:
        print("MISSING", m, e)
PY

echo "== 完成。重新打开 shell 或执行: source ~/.bashrc"
echo "   下一步: bash $MFP_ROOT/01_diffusion_inverse/scripts/00_download_totalseg.sh"
