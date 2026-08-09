# !/bin/bash


nohup python3 -m phase2.lightgbm.inference \
  --input /home/linux/7T/lzw/datasets/android_zoo/android_static \
  --model-dir runs/lightgbm/07-04_12-58_best/models \
  --output inference/lightgbm \
  --workers 12 > inference/lightgbm/lightgbm.log 2>&1 &