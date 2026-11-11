# 删除错误 APK 及提取结果

目录约定与 phase1 一致：三个根目录下按 `train|val|test`、`mal|benign`、APK 名称组织。脚本删除指定 APK 文件、对应的静态报告目录和图目录，并从相应划分的 `index.csv` 移除图索引行。默认仅预览；只有加 `--apply` 才修改数据。

单一划分、单一标签下清理多个 APK：

```bash
python -m deduplicate.remove_bad_apks \
  --apk-root /data/android_apks \
  --static-root /data/android_static \
  --graph-root /data/android_graphs \
  --split test --label benign \
  --sample-id app.one app.two.apk
```

同时清理不同划分或标签：准备含 `split,label,apk_name` 三列的 CSV，再使用 `--manifest bad_apks.csv`。标签可写 `mal`/`benign` 或 `1`/`0`。检查预览路径后，在相同命令末尾加 `--apply`；可用 `--audit-dir` 指定全新审计目录。执行时会先备份受影响的图索引及目标清单，原始 APK 不会备份。已生成的预测和解释结果需重新运行相应流程更新。
