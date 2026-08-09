#!/bin/sh

LOGDIR="runs/image/$(date +%m-%d_%H-%M)"
LOG="$LOGDIR/train.log" # 月日+时分秒
mkdir -p "$LOGDIR"  
CODEDIR="$LOGDIR/code"
mkdir -p "$CODEDIR"
cp train_image.sh "$CODEDIR"

setsid python3 -u -m phase2.image.train \
  --input "/home/linux/7T/lzw/datasets/android_zoo/android_dex_images" \
  --output "$LOGDIR" \
  --arch "convnextv2_nano" \
  --image-size 512 512 \
  --batch-size 32 \
  --epochs 200 \
  --class-weight \
  --info "使用convnextv2模型训练,尝试稍微增大模型参数" \
  --workers 4 > "$LOG" 2>&1 &

echo $$