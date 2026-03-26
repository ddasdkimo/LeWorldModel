#!/bin/bash
# 同步本地 LeWorldModel 專案到 DGX (GB10)
# 用法: ./scripts/sync-to-dgx.sh

DGX_HOST="rai@192.168.68.66"
DGX_PATH="/home/rai/code/LeWorldModel"
LOCAL_PATH="$(cd "$(dirname "$0")/.." && pwd)"

echo "📁 同步 $LOCAL_PATH → $DGX_HOST:$DGX_PATH"

rsync -avz --progress \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.venv' \
  --exclude 'venv' \
  --exclude '*.h5' \
  --exclude '*.hdf5' \
  --exclude '*.pt' \
  --exclude '*.pth' \
  --exclude '*.onnx' \
  --exclude 'wandb' \
  --exclude 'runs' \
  --exclude '.DS_Store' \
  "$LOCAL_PATH/" "$DGX_HOST:$DGX_PATH/"

echo "✅ 同步完成"
echo ""
echo "在 DGX 上執行:"
echo "  ssh $DGX_HOST"
echo "  source ~/code/lewm-env/bin/activate"
echo "  cd ~/code/LeWorldModel"
