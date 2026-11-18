# !/bin/bash


nohup python3 -m phase2.graph.inference \
  --input /home/linux/7T/lzw/datasets/android_zoo/android_graphs \
  --model-path runs/graph/07-03_15-37_best/models/gatv2_model.pt \
  --output inference/graph \
  --workers 20 > inference/graph/graph.log 2>&1 &