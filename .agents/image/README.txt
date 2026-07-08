学习率保持1e-3
batch size 32

num_masks不用太大，32即可
模型直接使用6e即可
对比流损失的权重一定要在0.001及以下
可能要测试一下batch_size的影响
不使用先验可达到相应的效果，符合思想



最佳结果（均基于不使用先验的情况）:
Confusion Matrix:
[[771  13]
 [ 32 164]]

Classification Report:
              precision    recall  f1-score   support

           0     0.9601    0.9834    0.9716       784
           1     0.9266    0.8367    0.8794       196

    accuracy                         0.9541       980
   macro avg     0.9434    0.9101    0.9255       980
weighted avg     0.9534    0.9541    0.9532       980


[Final Metrics]
ACC      : 95.41
ROC AUC  : 97.49
PR AUC   : 94.93 !
ACSA     : 91.01
GM       : 90.71


使用wandb进行超参数搜索
1.wandb sweep sweep.yaml # 向wandb录入超参数搜索的配置
2.wandb agent wandb_name/project_name/run_id # 运行超参数搜索的实验,已经集成在sweep_train.sh中


多模态集成接入

图像推理请直接读取 dex_images/test:

python phase2/image/classify.py \
  --data_path outputs/dex_images \
  --belong test \
  --eval_ckpt runs/image_model/best_final_model.pth \
  --output runs/image_model

推理输出:

- runs/image_model/predictions/test_predictions.csv
- runs/image_model/metrics/test_metrics.json
- runs/image_model/reports/test_confusion_matrix.csv
- runs/image_model/reports/test_classification_report.csv

predictions.csv 字段为:

sample_key,sample_id,apk_name,split,y_true,prob_malware,pred,image_path

其中 sample_key 格式为 split:label:sample_id，例如 test:1:sample001。该文件可直接传给 phase2/ensemble_predict.py:

python phase2/ensemble_predict.py \
  --input static=runs/static_lgbm/predictions/test_predictions.csv \
  --input graph=runs/hetero_hgt/predictions/test_predictions.csv \
  --input image=runs/image_model/predictions/test_predictions.csv \
  --output runs/ensemble
