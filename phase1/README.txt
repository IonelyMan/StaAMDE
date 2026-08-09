phase1: 数据准备的几个脚本，分别有静态提取和图构建

1. static_feature_extract.py
   批量提取 APK 静态文本特征，输出 static_features_report.json。保留Json文件

outputs/static_reports/
  mal/
    sample_a/
      static_features_report.json
  benign/
    sample_b/
      static_features_report.json

2. dex_to_rgb.py
   批量将 APK 内的 classes*.dex 转为 RGB 图像，供图像模态分类器使用。保留rgb图像。
   默认输出固定 256x256 真 RGB 图像，原始字节、相邻字节差分、错位字节分别组成 R/G/B 通道。
outputs/dex_images/
  train/mal/
    sample_a.png
  train/benign/
    sample_b.png

推荐命令：

python phase1/dex_to_rgb.py --dataset-root data/apks --belong train --output outputs/dex_images --shape-mode fixed_square --target-size 256 --color-mode rgb

如果需要复现旧版长图，可显式使用：

python phase1/dex_to_rgb.py --dataset-root data/apks --belong train --output outputs/dex_images_raw --shape-mode raw --color-mode viridis


3. build_heterogeneous.py
   基于 Manifest 与 DEX 调用关系构建 PyTorch Geometric HeteroData 异构图。保留生成的异构图
outputs/hetero_graphs/
  mal/
    sample_a/
      hetero_graph.pt
      hetero_graph_meta.json
  benign/
    sample_b/
      hetero_graph.pt
      hetero_graph_meta.json
  index.csv


三个脚本都支持：
- --belong train|val|test
- --dataset-root 原始数据根目录
- --input 1=恶意APK目录
- --input 0=良性APK目录
- --manifest apk_path,label CSV，可选 split 或 belong 列

如果原始数据目录为：

data/apks/train/mal
data/apks/train/benign
data/apks/val/mal
data/apks/val/benign
data/apks/test/mal
data/apks/test/benign

则可以用 --dataset-root data/apks --belong train 自动定位当前划分。


运行脚本


一、生成静态分析报告
只需要改 `--belong` 即可处理不同划分：

nohup python phase1/static_feature_extract.py \
  --dataset-root /home/linux/7T/lzw/datasets/android_zoo/android_apks \
  --belong test \
  --output /home/linux/7T/lzw/datasets/android_zoo/android_static \
  --workers 20 > extract.log 2>&1 &

python phase1/static_feature_extract.py --dataset-root data/apks --belong val --output outputs/static_reports --workers 8
python phase1/static_feature_extract.py --dataset-root data/apks --belong test --output outputs/static_reports --workers 8

也可以显式传目录：

python phase1/static_feature_extract.py ^
  --belong train ^
  --input 1=data/apks/train/mal ^
  --input 0=data/apks/train/benign ^
  --output outputs/static_reports

manifest CSV 也支持 split 或 belong：

apk_path,label,split
D:/dataset/train/mal/a.apk,1,train
D:/dataset/val/benign/b.apk,0,val
D:/dataset/test/mal/c.apk,1,test

python phase1/static_feature_extract.py --manifest data/manifest.csv --output outputs/static_reports


二、生成 DEX RGB 图像

python phase1/dex_to_rgb.py \
  --dataset-root /home/linux/7T/lzw/datasets/android_zoo/android_apks \
  --belong test \
  --output /home/linux/7T/lzw/datasets/android_zoo/android_dex_images \
  --source-mode structured \
  --shape-mode fixed_square \
  --target-size 512 \
  --color-mode rgb \
  --workers 30 \
  --overwrite

按同样方式把 `--belong` 换成 `val` 和 `test` 即可。

可选图像模式：
--source-mode structured  推荐，DEX结构感知，索引区 + data/code 区
--source-mode indexes     接近你 Java 代码，只用 header + DEX 索引表
--source-mode full        完整 DEX 字节流，信息最多但噪声最大
- `--color-mode rgb`：推荐默认值，原始字节、相邻字节差分、错位字节分别组成 R/G/B 通道。
- `--color-mode viridis/jet`：可视化更明显，但可能引入伪彩色噪声。

三、构建异构图

nohup python phase1/build_heterogeneous.py \
  --dataset-root /home/linux/7T/lzw/datasets/android_zoo/android_apks \
  --belong test \
  --output /home/linux/7T/lzw/datasets/android_zoo/android_graphs \
  --workers 30 > extract_graph.log 2>&1 &

按同样方式把 `--belong` 换成 `val` 和 `test`。默认索引会分别写到：

- outputs/hetero_graphs/train/index.csv
- outputs/hetero_graphs/val/index.csv
- outputs/hetero_graphs/test/index.csv