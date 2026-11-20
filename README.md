# StaAMDE：基于静态分析的安卓恶意软件检测与证据约束解释框架
- 李祖旺 徐志贤 吕轶黉 王炜
- 云南大学软件学院 昆明 650504
- （12025219065@stu.ynu.edu.cn）

**StaAMDE: A Static Analysis-Based Framework for Android Malware Detection and Evidence-Constrained Explanations**

Abstract：As the Android application ecosystem continues to evolve, malware detection is becoming increasingly challenging. Existing detection methods are prone to performance degradation under conditions of data class imbalance and time variation, while traditional discriminative evidence also suffers from limited user-friendliness. To address these issues, this paper proposes StaAMDE, a static analysis-based framework for Android malware detection and evidence-constrained explanations. In Stage 1, a dual-branch framework is employed to jointly model the static semantic features of APKs and their heterogeneous graph structures for classification while generating decision evidence. For the graph structure branch, we design ATGATv2, which uses node-type embeddings to retain type information for different program entities and aggregates neighborhood information using GATv2, thereby capturing key attention-based evidence while preserving the distinctions between heterogeneous entities. In Stage 2, attribution evidence from the two branches is integrated to constrain the LLM in generating explanation reports. Compared with the baselines, StaAMDE achieves better overall performance under class imbalance and temporal distribution shifts, demonstrating its effectiveness in improving Android malware detection performance. This framework decouples classification decisions from semantic explanation, making decision evidence easier to understand while improving detection performance. The source code is available at [https://github.com/IonelyMan/StaAMDE.git](https://github.com/IonelyMan/EEAMD.git).
Keywords：Android malware, User-friendliness, Static analysis, Graph attention networks, Large language models


## 目前在投，请勿抄袭，遵守学术诚信，以下文档教程基于AI总结，若有不足之处，敬请联系作者解决。

## 安装

在项目根目录执行以下命令。PyTorch 和 PyTorch Geometric 的版本需与运行环境的 CUDA 配置匹配。

```bash
python -m pip install -r requirements.txt
```

## 数据准备

项目提供实验所用的 4 个数据集，下载地址：[百度网盘](https://pan.baidu.com/s/1IBr__zLv5gxPVheBADBXhA)。由于 APK 发行版权和恶意 APK 传播限制，复现所需提取码可通过 12025219065@stu.ynu.edu.cn 联系作者获取。

下文所有命令均从项目根目录执行，路径均为示例相对路径。先将 APK 按已有的训练、验证和测试划分放置：

```text
data/apks/
  train/
    mal/*.apk
    benign/*.apk
  val/
    mal/*.apk
    benign/*.apk
  test/
    mal/*.apk
    benign/*.apk
```

`mal` 表示恶意，`benign` 表示良性。使用 `--belong` 分别提取三个划分；每条命令的输出根目录保持一致：

```bash
python phase1/static_feature_extract.py --dataset-root data/apks --belong train --output data/android_static --workers 8
python phase1/static_feature_extract.py --dataset-root data/apks --belong val   --output data/android_static --workers 8
python phase1/static_feature_extract.py --dataset-root data/apks --belong test  --output data/android_static --workers 8

python phase1/build_heterogeneous.py --dataset-root data/apks --belong train --output data/android_graphs --workers 8
python phase1/build_heterogeneous.py --dataset-root data/apks --belong val   --output data/android_graphs --workers 8
python phase1/build_heterogeneous.py --dataset-root data/apks --belong test  --output data/android_graphs --workers 8
```

静态报告写入 `data/android_static/<split>/<label>/<sample>/static_features_report.json`；图文件写入 `data/android_graphs/<split>/<label>/<sample>/hetero_graph.pt`，每个图划分还会生成 `index.csv`。图索引记录了生成时的图路径；移动图目录后应重新生成索引，避免推理和解释读取旧路径。

## 模型训练

两个分支读取上述输出目录。已有 `train/` 和 `val/` 划分时，训练沿用这两个划分；测试集留给最终评估。下例使用 `balanced` 类别权重。如需无权重对照，LightGBM 训练命令省略 `--class-weight balanced`，图训练命令改为 `--class-weight none`。

```bash
python -m phase2.lightgbm.train --input data/android_static --output runs/lightgbm --mi-k 5000 --workers 4 --class-weight balanced
python -m phase2.graph.train --input data/android_graphs --output runs/atgatv2 --model-type atgatv2 --workers 4 --class-weight balanced
```

训练完成后，LightGBM 模型位于 `runs/lightgbm/models/`，ATGATv2 检查点位于 `runs/atgatv2/models/atgatv2_model.pt`。两者的训练指标分别位于各自运行目录的 `metrics/` 中；图模型按验证集 PR-AUC 选择最佳轮次。

## 推理与集成

先分别生成验证集和测试集预测。下例保持分类阈值为 `0.5`：

```bash
python -m phase2.lightgbm.inference --input data/android_static --model-dir runs/lightgbm/models --output inference/lightgbm --split val --threshold 0.5
python -m phase2.graph.inference --input data/android_graphs --model-path runs/atgatv2/models/atgatv2_model.pt --output inference/atgatv2 --split val --threshold 0.5

python -m phase2.lightgbm.inference --input data/android_static --model-dir runs/lightgbm/models --output inference/lightgbm --split test --threshold 0.5
python -m phase2.graph.inference --input data/android_graphs --model-path runs/atgatv2/models/atgatv2_model.pt --output inference/atgatv2 --split test --threshold 0.5
```

只在验证集上搜索集成权重，再将该权重用于测试集；以下命令不搜索阈值：

```bash
python ensemble_predict.py \
  --input static=inference/lightgbm/predictions/test_predictions.csv \
  --input graph=inference/atgatv2/predictions/test_predictions.csv \
  --tune-input static=inference/lightgbm/predictions/val_predictions.csv \
  --tune-input graph=inference/atgatv2/predictions/val_predictions.csv \
  --tune-weights --threshold 0.5 --output ensemble
```

测试集结果位于 `ensemble/ensemble_predictions.csv`。集成器按 `sample_key`（划分、标签、样本 ID）对齐两个分支，未对齐的样本会被排除。若不调权重，省略两个 `--tune-input` 和 `--tune-weights`，默认使用等权集成。

## 单个 APK 的机器学习解释

以下使用虚构的测试集样本 `example.app.apk` 演示；将 `example.app` 换成实际 APK 文件名去掉 `.apk` 后的部分。先生成 LightGBM 的局部 SHAP 和 ATGATv2 的节点、边注意力：

```bash
python -m phase2.lightgbm.explain \
  --input data/android_static --model-dir runs/lightgbm/models \
  --split test --sample-id example.app --output ml_explain

python -m phase2.graph.explain \
  --input data/android_graphs --model-path runs/atgatv2/models/atgatv2_model.pt \
  --split test --sample-id example.app --output ml_explain
```

两个命令写入同一个样本目录：

```text
ml_explain/example.app/
  shap_contributions.csv
  shap_waterfall.png
  node_attention.csv
  edge_attention.csv
```

这些 CSV 是候选证据，尚未按最终传给 LLM 的条数裁剪。全局 SHAP 可单独计算，不会代替单个 APK 的局部证据：

```bash
python -m phase2.lightgbm.explain \
  --input data/android_static --model-dir runs/lightgbm/models \
  --split test --global --max-samples 1000 --output ml_explain
```

全局结果位于 `ml_explain/global/lightgbm/`。图解释如需处理当前划分的所有 APK，可将单样本命令中的 `--sample-id example.app` 改为 `--all`。

## 证据组织与 LLM 解释

运行解释代码前，需要配置LLM的API调用借口，位于config.py内。

第三阶段对同一 APK 的预测和局部解释进行证据组织。静态分支取 **SHAP 值最大的 15 个正向特征**和 **SHAP 值最小的 5 个负向特征**，分别保留将预测推向恶意和良性的线索。图分支先从节点候选中剔除 `app` 节点、从边候选中剔除自环，再按注意力分数取最高的 **15 个节点**和 **10 条边**；每条边证据包含源节点、目标节点及其注意力分数。过滤发生在证据组织阶段，原始注意力 CSV 仍保留应用节点和自环。可用条目不足时按实际数量保留。注意力表示图模型聚合信息时的关注线索，不单独证明恶意行为。

只组织证据和提示词、不调用 LLM 时，使用：

```bash
python -m phase3.llm_evidence \
  --sample-id example.app --split test \
  --static-predictions inference/lightgbm/predictions/test_predictions.csv \
  --graph-predictions inference/atgatv2/predictions/test_predictions.csv \
  --ensemble-predictions ensemble/ensemble_predictions.csv \
  --ml-explain-dir ml_explain --output llm_explain \
  --top-static-positive 15 --top-static-negative 5 \
  --top-graph-nodes 15 --top-graph-edges 10
```

在 `config.py` 配置调用所需的 `DASHSCOPE_API_KEY`、`DEFAULT_BASE_URL` 和 `DEFAULT_MODEL` 后，使用同样的证据参数生成解释报告：

```bash
python -m phase3.aliyun_llm \
  --sample-id example.app --split test \
  --static-predictions inference/lightgbm/predictions/test_predictions.csv \
  --graph-predictions inference/atgatv2/predictions/test_predictions.csv \
  --ensemble-predictions ensemble/ensemble_predictions.csv \
  --ml-explain-dir ml_explain --output llm_explain \
  --top-static-positive 15 --top-static-negative 5 \
  --top-graph-nodes 15 --top-graph-edges 10
```

`llm_explain/example.app/` 中的 `llm_evidence.json` 是筛选后的结构化证据，`llm_prompt.md` 是发送给模型的提示词，`llm_response.md` 是生成的报告。需要先检查证据而暂不调用 API 时，可在正式命令末尾添加 `--dry-run`。提示词会要求具备检索能力的模型优先核对 Android 官方文档。