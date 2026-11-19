# !/bin/bash


  nohup python3 -m phase2.graph.inference \
    --input /root/autodl-tmp/lzw/datasets/2_1/android_graphs \
    --model-path runs/graph/11-19_09-11_2_1/models/atgatv2_model.pt \
    --output inference/2_1/atgatv2 \
    --split test \
    --workers 4 > inference/2_1/atgatv2/graph.log 2>&1 &