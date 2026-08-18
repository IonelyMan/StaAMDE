# 《EEAMD：基于集成学习的安卓恶意软件检测与证据约束解释框架》
## EEAMD: Android malware detection and evidence constraint interpretation framework based on ensemble learning

EEAMD项目源码，论文已处于在投状态，请勿抄袭核心架构，务必遵守学术诚信。

提出一种基于集成学习的安卓恶意软件检测与证据约束解释框架 EEAMD。首先，构建覆盖 2020 至2026年、包含9855个样本的时间序列APK数据集，以贴近真实的演化环境。其次，设计集成分类架构：由 LightGBM 学习静态语义，提出TA-SGATv2学习图结构语义，在 GATv2 中首次引入节点类型嵌入以保留融合异构语义，从而提升检测性能。最后，提取SHAP贡献与图注意力作为结构化证据，约束LLM生成自然语言解释报告。与基线方法对比，该方法的F1 分数达0.8602，较次优基线提升 %5.8以上，几何平均值（G-mean）和平均类特定准确率（ACSA）分别提高 %4.9、%5.1 以上，表明该方法能够有效改善安卓恶意软件检测性能，通过机器学习与LLM解释的结合，在兼顾检测速度的基础上，显著增强了判别依据的可读性。

**项目教程：**
以下项目教程为使用AI工具总结，可能有误，望读者理清。


## 数据集下载
./zoo_android_csv_new 目录下提供了所有APK的sha256等信息，可以自行下载

一、模态组成

1. 静态文本特征
   - 提取 Permissions、Intents、Activities、Services、Receivers、Opcodes、API calls、API packages、System commands。
   - 额外派生 API families、Suspicious strings、Obfuscation 等安全语义特征。
   - 使用互信息筛选特征，再训练 LightGBM。
   - 若安装 shap，可单独输出 SHAP 全局解释结果。

2. DEX RGB 图像（已抛弃）
   - 从 APK 内提取 classes*.dex，默认转换为固定尺寸真 RGB 图像。
   - 默认每 3 个字节映射为 1 个 RGB 像素，不再输出灰度图。
   - 图像分类器由你的自创算法实现，只要最终导出统一格式 predictions.csv 即可进入集成。

3. Java/DEX 异构图
   - 基于 Androguard 从 DEX 和 Manifest 构建异构图。
   - 节点类型包括 app、class、method、api、package、permission、component、intent、opcode、api_family。
   - 使用 HGT 做 APK 级图分类。

4. 多模态集成
   - 各模态先独立训练、独立推理。
   - 集成阶段读取各模态 predictions.csv，按 sample_key 对齐概率。
   - 默认只集成各模态都预测到的样本，避免同名 APK、跨 split 样本或标签不同造成错配。

二、推荐目录

原始 APK：

```text
data/
  apks/
    train/
      mal/
      benign/
    val/
      mal/
      benign/
    test/
      mal/
      benign/
```

phase1 输出：

```text
outputs/
  static_reports/
    train/mal/sample_a/static_features_report.json
    val/benign/sample_b/static_features_report.json
    test/mal/sample_c/static_features_report.json
  dex_images/
    train/mal/sample_a.png
    val/benign/sample_b.png
    test/mal/sample_c.png
  hetero_graphs/
    train/mal/sample_a/hetero_graph_data.pt
    train/mal/sample_a/hetero_graph_meta.json
    train/index.csv
    val/index.csv
    test/index.csv
```

注意：`hetero_graph_data.pt` 是每个 APK 的 PyTorch Geometric 图数据样本，不是模型权重。真正的 HGT 模型权重保存为 `runs/hetero_hgt/models/hgt_model.pt`。

三、数据加载规则

训练阶段：

- LightGBM 训练只读取 `outputs/static_reports/train` 和 `outputs/static_reports/val`。
- TASGATv2 图模型训练只读取 `outputs/hetero_graphs/train` 和 `outputs/hetero_graphs/val`；根目录模式会根据目录名强制赋值 split，不依赖原始 JSON 或 index.csv 里的 split 字段。
- train/val 不要求三个模态样本完全一致，因为它们只是分别训练各自模型。
- 如果没有 train/val 子目录，代码才会回退到旧逻辑：扫描输入根目录并按已有 split 或随机切分。


推理阶段：

- 每个模态推理默认 `--split test`。
- 静态模态会优先直接扫描 `outputs/static_reports/test`。
- 图模态会优先直接扫描 `outputs/hetero_graphs/test`。
- 直接扫描 test 目录时，不再从每个 JSON 里读取 split 字段，速度更快。
- 图像模态也建议直接对 `outputs/dex_images/test` 推理，并导出 test 预测 CSV。

集成阶段：

- 集成器不再重新扫描原始数据或 split JSON。
- 它只读取各模态已经生成的 predictions.csv。
- 合并主键是 `sample_key`，格式为 `split:label:sample_id`，例如 `test:1:sample001`。
- 如果某个图像模型暂时没有输出 `sample_key`，至少要输出 `sample_id,split,y_true`，集成器会自动构造。

四、安装依赖

```bash
pip install -r requirements.txt
```

如果 torch-geometric 安装失败，请按你的 PyTorch 和 CUDA 版本参考官方安装命令安装。

五、训练静态文本分类器

```bash
python -m phase2.lightgbm.train \
  --input outputs/static_reports \
  --output runs/static_lgbm \
  --mi-k 5000 \
  --workers 4
```
互信息数量最好不要降低，保持5000

输出：

- `runs/static_lgbm/models/lightgbm_model.joblib`
- `runs/static_lgbm/models/dict_vectorizer.joblib`
- `runs/static_lgbm/models/selected_indices.npy`
- `runs/static_lgbm/metrics/train_metrics.json`
- `runs/static_lgbm/reports/mutual_info_features.csv`
- `runs/static_lgbm/reports/lightgbm_feature_importance.csv`

LightGBM early stopping 使用验证集 PR AUC 更新最佳模型。

推理：

```bash
python -m phase2.lightgbm.inference \
  --input outputs/static_reports \
  --model-dir runs/static_lgbm/models \
  --output runs/static_lgbm \
  --workers 4
```

默认等价于 `--split test`，会优先读取 `outputs/static_reports/test`。
`--workers` 控制静态 JSON 多进程加载数量；`--workers 0` 表示自动选择，机械硬盘建议 2-4，SSD 可尝试 8-12。

输出：

- `runs/static_lgbm/predictions/test_predictions.csv`
- `runs/static_lgbm/metrics/test_metrics.json`

推理指标包含 `accuracy`、`pr_auc`、`roc_auc`、`acsa`、`gm`、`recall`、`f1_score`、`confusion_matrix`、`classification_report`。

SHAP 解释：

```bash
python3 -m phase2.lightgbm.explain \
  --input /home/linux/7T/lzw/datasets/android_zoo/android_static \
  --model-dir runs/lightgbm/07-04_12-58_best/models \
  --output ml_explain/com.pcsensi.app \
  --split test \
  --waterfall-sample-id com.pcsensi.app
```
得到ml_explain/{id}/shap_paper_figures/{id}.png


六、训练异构图分类器

```bash
python -m phase2.graph.train \
  --input outputs/hetero_graphs \
  --output runs/hetero_gatv2 \
  --epochs 60 \
  --batch-size 8 \
  --model-type tasgatv2 \
  --workers 4
```

输出：

- `runs/hetero_hgt/models/hgt_model.pt`
- `runs/hetero_hgt/metrics/train_metrics.json`
- `runs/hetero_hgt/logs/history.json`

HGT 训练每轮使用验证集 PR AUC 判断并保存最佳模型权重。
`--workers` 会同时加速图元数据扫描和训练 DataLoader 读取图文件；`--workers 0` 表示自动选择。图文件较大时建议先从 2-4 开始，确认内存和磁盘压力正常后再尝试 8。

推理：

```bash
python3 -m phase2.graph.inference \
  --input /home/linux/7T/lzw/datasets/android_zoo/android_graphs \
  --model-path runs/graph/08-04_11-16_zero/models/gatv2_model.pt \
  --output inference/tasgatv2 \
  --split test
```

默认等价于 `--split test`，会优先读取 `outputs/hetero_graphs/test`。

输出：

- `runs/hetero_hgt/predictions/test_predictions.csv`
- `runs/hetero_hgt/metrics/test_metrics.json`

推理指标包含 `accuracy`、`pr_auc`、`roc_auc`、`acsa`、`gm`、`recall`、`f1_score`、`confusion_matrix`、`classification_report`。

可解释(只获取边和节点注意力的CSV文件)
python3 -m phase2.graph.explain \
  --input /home/linux/7T/lzw/datasets/android_zoo/android_graphs \
  --model-path runs/graph/07-03_15-37_best/models/gatv2_model.pt \
  --output ./ml_explain \
  --split test \
  --sample-id com.pcsensi.app

它会精确生成：
runs/hetero_gatv2/explanations/node_attention/{id}.csv
runs/hetero_gatv2/explanations/edge_attention/{id}.csv

# 可视化节点和边注意力
explain-dir目录下要包含egde_attention和node_attention目录(即对应样本其对应的节点/边注意力权重)
# 默认用这个,不支持多样本，要一个个生成图
```bash
python3 -m phase2.graph.visualize_attention \
  --explain-dir ml_explain/graph/explanations \
  --top-k-nodes 30 \
  --neighbor-hops 1 \
  --max-display-nodes 60 \
  --max-display-edges 180 \
  --circular-order spread \
  --title "TASGATv2-com.pcsensi.app" \
  --sample-id com.pcsensi.app \
  --output-dir ml_explain
```

产出ml_explain/com.pcsensi.app/attention_paper_figures/com.pcsensi.app.png

七、图像模态接入（已抛弃）

图像模态代码位于 `phase2/image`，当前分类模型为 ConvNeXt V2。训练按 `outputs/dex_images/train` 和 `outputs/dex_images/val` 加载图像数据；推理默认按 `outputs/dex_images/test` 输出统一 predictions.csv。
支持模型规格：
```text
convnextv2_atto, convnextv2_femto, convnextv2_pico, convnextv2_nano,
convnextv2_tiny, convnextv2_base, convnextv2_large
```
训练：
```bash
python -m phase2.image.train \
  --input outputs/dex_images \
  --output runs/image_convnextv2 \
  --arch convnextv2_atto \
  --image-size 224 224 \
  --batch-size 32 \
  --epochs 100

输出：

- `runs/image_model/predictions/test_predictions.csv`
- `runs/image_model/metrics/test_metrics.json`
- `runs/image_model/reports/test_confusion_matrix.csv`
- `runs/image_model/reports/test_classification_report.csv`

图像模态的 predictions.csv 已经使用统一格式，可直接集成：

```csv
sample_key,sample_id,apk_name,split,y_true,prob_malware,pred
test:1:sample001,sample001,sample001.apk,test,1,0.932,1
```

八、多模态概率集成

只有静态和图模型：

```bash
python ensemble_predict.py \
  --input static=inference/lightgbm/predictions/test_predictions.csv \
  --input graph=inference/tasgatv2/predictions/test_predictions.csv \
  --output ensemble \
  --tune-weights \
  --tune-threshold
```

加入图像模型后：（已抛弃）

```bash
python ensemble_predict.py \
  --input static=inference/lightgbm/predictions/test_predictions.csv \
  --input graph=inference/graph/predictions/test_predictions.csv \
  --input image=inference/image/predictions/test_predictions.csv \
  --output ensemble \
  --tune-weights \
  --tune-threshold
```

输出：

- `runs/ensemble/ensemble_predictions.csv`
- `runs/ensemble/ensemble_metrics.json`
- `runs/ensemble/ensemble_confusion_matrix.csv`
- `runs/ensemble/ensemble_classification_report.csv`

集成指标包含 `accuracy`、`pr_auc`、`roc_auc`、`acsa`、`gm`、`recall`、`f1_score`。如果输入 CSV 中包含 split，还会额外输出各 split 的混淆矩阵和分类报告。

九、实验建议

1. 先每个 split 跑少量样本做端到端冒烟实验。
2. 正式跑 1 万 APK 时，phase1 脚本默认会跳过已有结果，适合分批跑。
3. train/val 阶段不要强行要求三模态样本一致，先让每个模态充分利用自己的可用数据。
4. 最终报告 test 指标时，用各模态 test 预测 CSV 做集成，并明确说明集成样本数是各模态 sample_key 的交集。
