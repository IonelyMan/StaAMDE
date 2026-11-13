# !/bin/bash


nohup python3 -m phase2.lightgbm.inference \
  --input /root/autodl-tmp/lzw/datasets/StaADME_dataset1/android_static \
  --model-dir runs/lightgbm/11-12_22-35/models \
  --output inference/lightgbm \
  --split val \
  --workers 12 > inference/lightgbm/lightgbm.log 2>&1 &