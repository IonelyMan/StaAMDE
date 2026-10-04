phase1 负责从 APK 提取两类输入：

1. static_feature_extract.py：保存静态特征报告 static_features_report.json。
2. build_heterogeneous.py：保存异构图和节点元数据。

两个脚本均支持 --dataset-root、--belong train|val|test、--input 和 --manifest。

示例：
python phase1/static_feature_extract.py --dataset-root data/apks --belong test --output outputs/static_reports --workers 8
python phase1/build_heterogeneous.py --dataset-root data/apks --belong test --output outputs/hetero_graphs --workers 8
