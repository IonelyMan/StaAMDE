# !/bin/bash

python ensemble_predict.py \
  --input static=inference/lightgbm/predictions/test_predictions.csv \
  --input graph=inference/graph/predictions/test_predictions.csv \
  --input image=inference/image/predictions/test_predictions.csv \
  --output ensemble \
  --tune-weights \
  --tune-threshold