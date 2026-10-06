# SAMDE：基于静态分析的安卓恶意软件检测与证据约束解释框架

**SAMDE: A Static Analysis-Based Framework for Android Malware Detection and Evidence-Constrained Interpretation**

SAMDE 由两条静态分析分类分支组成：LightGBM 学习 APK 静态特征，ATGATv2（**A**ndroid Node **T**ype GATv2）学习从 APK 构建的图结构。集成器按 `sample_key` 对齐两者的预测概率。解释阶段保存 LightGBM 的 SHAP 贡献和图模型的节点、边注意力，再把结构化证据交给 LLM。

## 安装

```bash
pip install -r requirements.txt
```

PyTorch Geometric 的安装方式需与本机 PyTorch/CUDA 版本匹配。

## 数据准备与训练

原始 APK 可按 `data/apks/{train,val,test}/{mal,benign}/` 放置。

```bash
python phase1/static_feature_extract.py --dataset-root data/apks --belong train --output /home/linux/7T/lzw/datasets/android_zoo/android_static --workers 8
python phase1/build_heterogeneous.py --dataset-root data/apks --belong train --output ~/7T/lzw/datasets/android_zoo/android_graphs --workers 8
python -m phase2.lightgbm.train --input /home/linux/7T/lzw/datasets/android_zoo/android_static --output runs/static_lgbm --mi-k 5000 --workers 4
python -m phase2.graph.train --input ~/7T/lzw/datasets/android_zoo/android_graphs --output runs/hetero_gatv2 --model-type atgatv2 --workers 4
```

对 `val` 和 `test` 重复前两条提取命令。静态报告与异构图分别由各自模型读取。

## 推理与集成

```bash
python -m phase2.lightgbm.inference --input /home/linux/7T/lzw/datasets/android_zoo/android_static --model-dir runs/static_lgbm/models --output inference/lightgbm --split test
python -m phase2.graph.inference --input ~/7T/lzw/datasets/android_zoo/android_graphs --model-path runs/hetero_gatv2/models/gatv2_model.pt --output inference/atgatv2 --split test
python ensemble_predict.py --input static=inference/lightgbm/predictions/test_predictions.csv --input graph=inference/atgatv2/predictions/test_predictions.csv --output ensemble
```

模型类型参数现统一为 `atgatv2`，目录名不影响模型加载。

集成结果在 `ensemble/ensemble_predictions.csv`。输入 CSV 需要包含 `sample_key`，或者包含可构造该键的 `sample_id,split,y_true`。同一 APK 在两个分支的键必须一致。

## 机器学习解释

指定 APK 使用 `--sample-id`。两个脚本都支持一次传入多个 ID，也支持重复写 `--sample-id` 或用逗号分隔；结果分别保存在 APK 名称对应的小目录中。默认只解释 test 集。

```bash
python -m phase2.lightgbm.explain \
  --input /home/linux/7T/lzw/datasets/android_zoo/android_static \
  --model-dir runs/lightgbm/07-04_12-58_best/models \
  --split test --sample-id com.pcsensi.app org.ooma.oomaapp \
  --output ml_explain

python -m phase2.graph.explain --input ~/7T/lzw/datasets/android_zoo/android_graphs \
  --model-path runs/graph/07-03_15-37_best/models/gatv2_model.pt \
  --split test --sample-id com.pcsensi.app org.ooma.oomaapp \
  --output ml_explain
```

以上命令写出：

```text
ml_explain/sample001/
  shap_contributions.csv
  shap_waterfall.png
  node_attention.csv
  edge_attention.csv
```

全局 SHAP 有独立入口，保存当前 split 的特征平均贡献摘要，以及重要性柱状图和摘要散点图；不会覆盖任何单个 APK 的证据：

```bash
python -m phase2.lightgbm.explain --input /home/linux/7T/lzw/datasets/android_zoo/android_static --model-dir runs/lightgbm/07-04_12-58_best/models --global --split test --max-samples 10000 --output ml_explain
```

输出在 `ml_explain/global/lightgbm/`：`test_shap_summary.csv`、`test_shap_mean_abs_bar.png`、`test_shap_summary.png`。可用 `--waterfall-top-k` 和 `--global-plot-top-k` 控制图中展示的特征数，`--figure-format` 选择 PNG、PDF 或 SVG。图模型使用 `--all` 可逐个输出某个 split 的所有 APK 的节点、边注意力 CSV；不绘制图结构证据。

## LLM 证据与解释

```bash
python -m phase3.llm_evidence --sample-id sample001 --ml-explain-dir ml_explain --static-predictions inference/lightgbm/predictions/test_predictions.csv --graph-predictions inference/tasgatv2/predictions/test_predictions.csv --ensemble-predictions ensemble/ensemble_predictions.csv --output llm_explain
```

结果保存在对应 APK 名称的小目录中的 `llm_evidence.json` 与 `llm_prompt.md`。传给 LLM 的证据 JSON 使用简短中文键名，只包含该 APK 的局部 SHAP 和图注意力；全局 SHAP 仍可独立计算和绘图，但不输入 LLM。提示词要求有联网能力的 LLM 优先核对 Android Developers 官方文档；实际使用的模型仍需具备联网工具才能检索。

可使用 `python -m phase3.aliyun_llm ... --dry-run` 仅生成证据与提示词。正式调用还需要自行配置该脚本使用的 API 参数。
