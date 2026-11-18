# !/bin/bash


nohup python3 -m phase2.lightgbm.inference \
  --input /home/linux/7T/lzw/datasets/CICMalDroid2020/2_1/android_static \
  --model-dir runs/lightgbm/11-13_11-04_2:1/models \
  --output inference/2_1/lightgbm \
  --split test \
  --workers 12 > inference/lightgbm/lightgbm.log 2>&1 &