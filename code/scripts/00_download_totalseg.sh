#!/usr/bin/env bash
# 下载 TotalSegmentator v2.0.1 CT 数据集（Zenodo 10047292, 23.6 GB, md5 fe250e5718e0a3b5df4c4ea9d58a62fe）
# 用法：
#   bash 00_download_totalseg.sh            # 全集
#   SUBSET_URL=<zenodo小子集zip链接> bash 00_download_totalseg.sh   # 102 例小子集（链接见 Zenodo 页面 "A small subset ... here"）
set -euo pipefail

DATA_ROOT="${MFP_DATA_ROOT:-/root/autodl-tmp/data}"
RAW="$DATA_ROOT/totalseg/raw"
mkdir -p "$RAW"
cd "$RAW"

if [ -n "${SUBSET_URL:-}" ]; then
  ZIP="totalseg_subset.zip"
  URL="$SUBSET_URL"
  MD5=""
else
  ZIP="Totalsegmentator_dataset_v201.zip"
  URL="https://zenodo.org/records/10047292/files/Totalsegmentator_dataset_v201.zip?download=1"
  MD5="fe250e5718e0a3b5df4c4ea9d58a62fe"
fi

if [ ! -f "$ZIP" ]; then
  echo "下载 $URL -> $RAW/$ZIP（Zenodo 直连可能较慢；页面另有 Dropbox 镜像可手动替换 URL）"
  # -c 断点续传
  wget -c -O "$ZIP" "$URL"
fi

if [ -n "$MD5" ]; then
  echo "校验 md5 ..."
  echo "$MD5  $ZIP" | md5sum -c -
fi

if [ ! -d "extracted" ]; then
  echo "解压 ..."
  mkdir -p extracted
  unzip -q "$ZIP" -d extracted
fi

echo "目录结构示例："
find extracted -maxdepth 2 | head -20
echo "meta.csv 位置："
find extracted -name meta.csv | head -1
echo "完成。下一步：python src/data/prepare_slices.py --raw $RAW/extracted --out $DATA_ROOT/totalseg/processed"
