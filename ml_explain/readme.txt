机器学习解释的统一根目录。

ml_explain/<APK文件名去掉.apk>/
  shap_contributions.csv    单个 APK 的 LightGBM 局部 SHAP
  shap_waterfall.png         单个 APK 的 SHAP 瀑布图
  node_attention.csv         单个 APK 的图节点注意力
  edge_attention.csv         单个 APK 的图边注意力

ml_explain/global/lightgbm/<split>_shap_summary.csv
  --global 入口生成的全局 SHAP 摘要
ml_explain/global/lightgbm/<split>_shap_mean_abs_bar.png
  全局平均绝对 SHAP 重要性柱状图
ml_explain/global/lightgbm/<split>_shap_summary.png
  全局 SHAP 摘要散点图

三个局部 CSV 可直接由 phase3.llm_evidence 读取。已有的旧目录是历史示例结果，新运行使用上述布局。
