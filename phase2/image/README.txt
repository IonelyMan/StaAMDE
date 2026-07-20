ConvNeXt V2 图像模态

数据目录要求:

outputs/dex_images/
  train/mal/*.png
  train/benign/*.png
  val/mal/*.png
  val/benign/*.png
  test/mal/*.png
  test/benign/*.png

支持类别目录别名:

- 恶意: mal, malware, 1
- 良性: benign, leg, normal, 0

支持模型:

- convnextv2_atto
- convnextv2_femto
- convnextv2_pico
- convnextv2_nano
- convnextv2_tiny
- convnextv2_base
- convnextv2_large

训练:

python -m phase2.image.train \
  --input outputs/dex_images \
  --output runs/image_convnextv2 \
  --arch convnextv2_atto \
  --image-size 224 224 \
  --batch-size 32 \
  --epochs 100

推理:

python -m phase2.image.inference \
  --input outputs/dex_images \
  --model-path runs/image_convnextv2/models/best_model.pt \
  --output runs/image_convnextv2 \
  --split test

输出:

- runs/image_convnextv2/predictions/test_predictions.csv
- runs/image_convnextv2/metrics/test_metrics.json
- runs/image_convnextv2/reports/test_confusion_matrix.csv
- runs/image_convnextv2/reports/test_classification_report.csv

predictions.csv 字段:

sample_key,sample_id,apk_name,split,y_true,prob_malware,pred,image_path

该文件可直接进入多模态集成:

python phase2/ensemble_predict.py \
  --input static=runs/static_lgbm/predictions/test_predictions.csv \
  --input graph=runs/hetero_gatv2/predictions/test_predictions.csv \
  --input image=runs/image_convnextv2/predictions/test_predictions.csv \
  --output runs/ensemble
