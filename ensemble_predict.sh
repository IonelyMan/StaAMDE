#!/bin/bash
python ensemble_predict.py \
  --input static=inference/lightgbm/predictions/test_predictions.csv \
  --input graph=inference/atgatv2/predictions/test_predictions.csv \
  --tune-input static=inference/lightgbm/predictions/val_predictions.csv \
  --tune-input graph=inference/atgatv2/predictions/val_predictions.csv \
  --output ensemble \
  --tune-weights \
  --threshold 0.5 \
