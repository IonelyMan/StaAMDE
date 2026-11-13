# 自动清理错误 APK 及提取结果

## 清理缺少图边索引的样本

在训练所用的 PyTorch、PyTorch Geometric 环境中，先扫描图文件并生成清单：

```bash
python -m deduplicate.find_missing_graph_edges \
  --graph-root /home/linux/7T/lzw/datasets/CICMalDroid2020/2_1/android_graphs \
  --split train --output missing_graph_edges.csv
```

扫描按图训练的同构转换检查 `edge_index is None`，只检查 `train/benign/<ID>/` 和 `train/mal/<ID>/` 下训练支持的图文件；省略 `--split` 可扫描 train、val、test。同一目录有两个图文件时按目录训练的优先级选择，使用自定义索引训练时路径可能不同。空边张量不是缺失索引，不会列入清单。扫描失败时需先排查错误，不能将失败样本自动视为待删除样本。

再使用同一清单预览清理，以下 APK 和静态数据路径应替换为实际路径：

```bash
python -m deduplicate.remove_bad_apks \
  --apk-root /实际路径/android_apks \
  --static-root /实际路径/android_static \
  --graph-root /home/linux/7T/lzw/datasets/CICMalDroid2020/2_1/android_graphs \
  --manifest missing_graph_edges.csv
```

检查预览后在命令末尾加 `--apply` 执行。工具要求 APK 位于 `APK根目录/split/label/ID.apk`，按 `split/label/ID` 删除对应 APK 和静态、图样本目录，并更新 `图根目录/split/index.csv`；会备份原图索引和清单，但不会备份被删除的 APK 或提取结果。仅支持以上目录结构，其他集中式表格或索引不会自动修改。图缺边可能由提取失败造成，如需保留 APK，请先检查或重新提取；此清理操作会删除 APK。

## 先排查数量差异

原始 APK 比 train 的提取结果多时，先运行只读对账脚本：

```bash
python -m deduplicate.audit_extraction \
  --apk-root /home/linux/7T/lzw/datasets/android_zoo/android_apks \
  --static-root /home/linux/7T/lzw/datasets/android_zoo/android_static \
  --graph-root /home/linux/7T/lzw/datasets/android_zoo/android_graphs \
  --split train
```

它分别列出 benign、mal 中缺失或报错的样本 ID，并检查 APK 是否为有效 ZIP、静态报告中的 `error`、图索引中的 `failed`/`error`。静态报告即使存在，只要含 `error`，LightGBM 加载器也会跳过它；仅凭目录数量可能看不出这一点。排查脚本不会删除数据。

## 清理明确失败的样本

脚本扫描 `train`、`val`、`test` 的 `mal` 和 `benign` 目录，自动识别三类明确错误：静态报告 `static_features_report.json` 含非空 `error`、图索引 `index.csv` 的 `status=failed` 或非空 `error`、APK 文件不是有效 ZIP。**仅缺少提取结果不会被判为错误。**清理时会删除原始 APK、对应静态报告目录和图目录，并从该划分的图索引移除记录。

默认扫描全部划分和标签，只预览，不修改数据：

```bash
python -m deduplicate.remove_bad_apks \
  --apk-root /home/linux/7T/lzw/datasets/android_zoo/android_apks \
  --static-root /home/linux/7T/lzw/datasets/android_zoo/android_static \
  --graph-root /home/linux/7T/lzw/datasets/android_zoo/android_graphs \
```

可加 `--split test` 或 `--label benign` 缩小扫描范围。检查预览中的错误原因和路径后，在相同命令末尾加 `--apply`。执行前会把目标清单和受影响的图索引备份到新的审计目录；可用 `--audit-dir` 指定该目录。原始 APK 不会备份。

`--sample-id` 和含 `split,label,apk_name` 三列的 `--manifest` 仍可用于需要手动指定目标的情况。已生成的预测与解释结果不会自动更新，需要重新运行相应流程。
