# !/bin/bash
# 抛弃图像模态
python ensemble_predict.py \
  --input static=inference/lightgbm/predictions/test_predictions.csv \
  --input graph=inference/ta-sgatv2/predictions/test_predictions.csv \
  --output ensemble \
  --tune-weights \
  --tune-threshold