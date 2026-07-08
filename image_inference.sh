#!/bin/sh

OUTDIR="inference/image"
mkdir -p "$OUTDIR"
LOG="$OUTDIR/classify.log"

nohup python3 phase2/image/inference.py \
    --gpu 0 \
    --batch_size 32 \
    --learning_rate 1e-3 \
    --num_attn_layers 2 \
    --num_classes 2 \
    --image_size 512 512 \
    --num_masks 32 \
    --latent_input_dim 8 \
    --data_path /home/linux/7T/lzw/datasets/android_zoo/android_dex_images \
    --eval_ckpt /home/linux/7T/lzw/projects/multimodel_malware_detection/runs/image/07-02_10-59/best_final_model.pth \
    --output "$OUTDIR" \
    --belong test \
    --info "" > "$LOG" 2>&1 &

echo $!
