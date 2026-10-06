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
python phase1/static_feature_extract.py --dataset-root data/apks --belong train --output outputs/static_reports --workers 8
python phase1/build_heterogeneous.py --dataset-root data/apks --belong train --output outputs/hetero_graphs --workers 8
python -m phase2.lightgbm.train --input outputs/static_reports --output runs/static_lgbm --mi-k 5000 --workers 4
python -m phase2.graph.train --input outputs/hetero_graphs --output runs/hetero_gatv2 --model-type atgatv2 --workers 4
```

对 `val` 和 `test` 重复前两条提取命令。静态报告与异构图分别由各自模型读取。

## 推理与集成

```bash
python -m phase2.lightgbm.inference --input outputs/static_reports --model-dir runs/static_lgbm/models --output inference/lightgbm --split test
python -m phase2.graph.inference --input outputs/hetero_graphs --model-path runs/hetero_gatv2/models/gatv2_model.pt --output inference/atgatv2 --split test
python ensemble_predict.py --input static=inference/lightgbm/predictions/test_predictions.csv --input graph=inference/atgatv2/predictions/test_predictions.csv --output ensemble
```

模型类型参数现统一为 `atgatv2`，目录名不影响模型加载。

集成结果在 `ensemble/ensemble_predictions.csv`。输入 CSV 需要包含 `sample_key`，或者包含可构造该键的 `sample_id,split,y_true`。同一 APK 在两个分支的键必须一致。

## 机器学习解释

单个 APK 的解释使用 `--sample-id`，两个脚本共用一个输出根目录。目录名取 APK 文件名去掉 `.apk` 后的名称。

```bash
python -m phase2.lightgbm.explain --input outputs/static_reports --model-dir runs/static_lgbm/models --sample-id sample001 --output ml_explain
python -m phase2.graph.explain --input outputs/hetero_graphs --model-path runs/hetero_gatv2/models/gatv2_model.pt --sample-id sample001 --output ml_explain
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
python -m phase2.lightgbm.explain --input outputs/static_reports --model-dir runs/static_lgbm/models --global --split test --max-samples 800 --output ml_explain
```

输出在 `ml_explain/global/lightgbm/`：`test_shap_summary.csv`、`test_shap_mean_abs_bar.png`、`test_shap_summary.png`。可用 `--waterfall-top-k` 和 `--global-plot-top-k` 控制图中展示的特征数，`--figure-format` 选择 PNG、PDF 或 SVG。图模型使用 `--all` 可逐个输出某个 split 的所有 APK 的节点、边注意力 CSV；不绘制图结构证据。

## LLM 证据与解释

```bash
python -m phase3.llm_evidence --sample-id sample001 --ml-explain-dir ml_explain --static-predictions inference/lightgbm/predictions/test_predictions.csv --graph-predictions inference/tasgatv2/predictions/test_predictions.csv --ensemble-predictions ensemble/ensemble_predictions.csv --output llm_explain
```

结果保存在 `llm_explain/sample001/llm_evidence.json` 与 `llm_prompt.md`。提示词要求有联网能力的 LLM 优先核对 Android Developers 官方文档，引用可核验来源，并标明无法核验之处。若实际调用的模型没有联网工具，提示词会要求明确说明这一限制；仅添加提示词不能赋予模型联网能力。

可使用 `python -m phase3.aliyun_llm ... --dry-run` 仅生成证据与提示词。正式调用还需要自行配置该脚本使用的 API 参数。
