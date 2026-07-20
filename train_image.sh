#!/bin/sh

LOGDIR="runs/image/$(date +%m-%d_%H-%M)"
LOG="$LOGDIR/train.log" # 月日+时分秒
mkdir -p "$LOGDIR"  
CODEDIR="$LOGDIR/code"
mkdir -p "$CODEDIR"
cp train_image.sh "$CODEDIR"

python -u -m phase2.image.train \
  --input "/home/linux/7T/lzw/datasets/android_zoo/android_dex_images" \
  --output "$LOGDIR" \
  --arch "atto" \
  --image-size 224 224 \
  --batch-size 32 \
  --epochs 100 \
  --workers 4 2>&1 | tee "$LOG"

echo $$