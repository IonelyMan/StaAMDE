# !/bin/bash
python ensemble_predict.py \
  --input static=inference/lightgbm/predictions/test_predictions.csv \
  --input graph=inference/atgatv2/predictions/test_predictions.csv \
  --output ensemble \
  --tune-weights \
  --tune-threshold
