# 自动清理错误 APK 及提取结果

脚本扫描 `train`、`val`、`test` 的 `mal` 和 `benign` 目录，自动识别三类明确错误：静态报告 `static_features_report.json` 含非空 `error`、图索引 `index.csv` 的 `status=failed` 或非空 `error`、APK 文件不是有效 ZIP。**仅缺少提取结果不会被判为错误。**清理时会删除原始 APK、对应静态报告目录和图目录，并从该划分的图索引移除记录。

默认扫描全部划分和标签，只预览，不修改数据：

```bash
python -m deduplicate.remove_bad_apks \
  --apk-root /data/android_apks \
  --static-root /data/android_static \
  --graph-root /data/android_graphs
```

可加 `--split test` 或 `--label benign` 缩小扫描范围。检查预览中的错误原因和路径后，在相同命令末尾加 `--apply`。执行前会把目标清单和受影响的图索引备份到新的审计目录；可用 `--audit-dir` 指定该目录。原始 APK 不会备份。

`--sample-id` 和含 `split,label,apk_name` 三列的 `--manifest` 仍可用于需要手动指定目标的情况。已生成的预测与解释结果不会自动更新，需要重新运行相应流程。
