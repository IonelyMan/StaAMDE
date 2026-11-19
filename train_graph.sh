# !/bin/bash

LOGDIR="runs/graph/$(date +%m-%d_%H-%M)_2_1"
LOG="$LOGDIR/train.log" # 月日+时分秒
mkdir -p "$LOGDIR"
INPUT_DIR="/root/autodl-tmp/lzw/datasets/2_1/android_graphs"

# 耗显存太大，只能调为8
nohup python3 -u ./phase2/graph/train.py \
  --batch-size 8 \
  --class-weight none \
  --input "$INPUT_DIR" \
  --model-type "atgatv2" \
  --workers 12 \
  --type-embedding-dim 16 \
  --info "恢复1.0 图训练逻辑，不加入类别权重" \
  --output "$LOGDIR" > "$LOG" 2>&1 &

echo $!

