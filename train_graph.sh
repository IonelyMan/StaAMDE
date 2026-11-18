# !/bin/bash

LOGDIR="runs/graph/$(date +%m-%d_%H-%M)"
LOG="$LOGDIR/train.log" # 月日+时分秒
mkdir -p "$LOGDIR"
INPUT_DIR="/home/linux/7T/lzw/datasets/android_zoo/android_graphs"

# 耗显存太大，只能调为8
nohup python3 -u ./phase2/graph/train.py \
  --batch-size 8 \
  --class-weight "${GRAPH_CLASS_WEIGHT:-balanced}" \
  --input "$INPUT_DIR" \
  --model-type "atgatv2" \
  --workers 12 \
  --type-embedding-dim 16 \
  --info "恢复1.0 图训练逻辑，类别权重模式: ${GRAPH_CLASS_WEIGHT:-balanced}" \
  --output "$LOGDIR" > "$LOG" 2>&1 &

echo $!

