#!/bin/sh

LOGDIR="runs/image/$(date +%m-%d_%H-%M)"
LOG="$LOGDIR/train.log" # 月日+时分秒
mkdir -p "$LOGDIR"  
CODEDIR="$LOGDIR/code"
mkdir -p "$CODEDIR"
cp train_image.sh "$CODEDIR"

nohup python3 phase2/image/train.py \
    --gpu 0 \
    --epochs 200 \
    --batch_size 64 \
    --learning_rate 1e-3 \
    --num_attn_layers 2 \
    --num_classes 2 \
    --latent_weight 1 \
    --flow_contrastive_weight 0.01 \
    --feature_contrastive_weight 1 \
    --ortho_weight 1 \
    --feature_margin 1 \
    --flow_margin 1 \
    --image_size 256 256 \
    --num_masks 32 \
    --latent_input_dim 8 \
    --save_dir "$LOGDIR" \
    --data_path /home/linux/7T/lzw/datasets/android_zoo/android_dex_images \
    --info "图像模态训练任务,降低图像大小" > "$LOG" 2>&1 &

echo $$