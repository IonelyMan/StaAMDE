# !/bin/bash

LOGDIR="runs/lightgbm/$(date +%m-%d_%H-%M)"
LOG="$LOGDIR/train.log" # 月日+时分秒
mkdir -p "$LOGDIR"
INPUT_DIR="/root/autodl-tmp/lzw/datasets/StaADME_dataset1/android_static"

nohup python3 -u phase2/lightgbm/train.py \
  --input "$INPUT_DIR" \
  --output "$LOGDIR" \
  --workers 12 \
  --info "训练打印信息" \
  --class-weight balanced \
  --mi-k 5000 > "$LOG" 2>&1 &
