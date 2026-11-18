#!/bin/bash
python ensemble_predict.py \
  --input static=inference/StaAMDE_dataset2/lightgbm/predictions/test_predictions.csv \
  --input graph=inference/StaAMDE_dataset2/atgatv2/predictions/test_predictions.csv \
  --tune-input static=inference/StaAMDE_dataset2/lightgbm/predictions/val_predictions.csv \
  --tune-input graph=inference/StaAMDE_dataset2/atgatv2/predictions/val_predictions.csv \
  --output ensemble/StaAMDE_dataset2 \
  --tune-weights \
  --tune-threshold
