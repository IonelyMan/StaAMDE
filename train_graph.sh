# !/bin/bash

LOGDIR="runs/graph/$(date +%m-%d_%H-%M)"
LOG="$LOGDIR/train.log" # 月日+时分秒
mkdir -p "$LOGDIR"
INPUT_DIR="/home/linux/7T/lzw/datasets/android_zoo/android_graphs"

# 耗显存太大，只能调为8
nohup python3 -u ./phase2/graph/train.py \
  --batch-size 8 \
  --input "$INPUT_DIR" \
  --hidden-channels 64 \
  --lr 0.0005 \
  --weight-decay 5e-4 \
  --workers 24 \
  --info "降低学习率，降低hiddenc,降低weight-decay" \
  --output "$LOGDIR" > "$LOG" 2>&1 &

echo $!

# python -m phase2.graph.inference \
#   --input "$INPUT_DIR" \
#   --model-path "$RUN_DIR/models/hgt_model.pt" \
#   --output "$RUN_DIR" \
#   --split all
