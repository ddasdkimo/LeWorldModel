#!/bin/bash
# 在 DGX 上遠端執行命令
# 用法: ./scripts/run-on-dgx.sh "python train.py data=pusht"

DGX_HOST="rai@192.168.68.66"
CMD="${1:-echo 'no command specified'}"

ssh -t "$DGX_HOST" "
  export PATH=\$HOME/.local/bin:\$PATH
  source ~/code/lewm-env/bin/activate
  cd ~/code/LeWorldModel
  $CMD
"
