import argparse
import copy
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import List, Optional

import numpy as np
from sklearn.feature_extraction import DictVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase2.common.metrics import binary_metrics
from phase2.lightgbm.dataset import StaticDataset, load_static_dataset
from phase2.lightgbm.model import load_artifacts, predict_probability
from phase2.lightgbm.reports import write_json, write_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用已训练 LightGBM 模型进行推理")
    parser.add_argument("--input", required=True, help="phase1/static_feature_extract.py 输出目录")
    parser.add_argument("--model-dir", required=True, help="训练得到的 models 目录")
    parser.add_argument("--output", required=True, help="推理输出目录，例如 runs/static_lgbm")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=4, help="静态 JSON 并行加载进程数；0 表示自动")
    return parser.parse_args()


def filter_indices(splits: List[Optional[str]], split: str) -> np.ndarray:
    if split == "all":
        return np.arange(len(splits))
    return np.asarray([idx for idx, value in enumerate(splits) if value == split], dtype=np.int64)


def resolve_split_input(input_dir: Path, split: str) -> Path:
    if split == "all":
        return input_dir
    split_dir = input_dir / split
    return split_dir if split_dir.exists() else input_dir


def subset(dataset: StaticDataset, indices: np.ndarray):
    return (
        [dataset.sample_keys[int(index)] for index in indices],
        [dataset.sample_ids[int(index)] for index in indices],
        [dataset.apk_names[int(index)] for index in indices],
        [dataset.splits[int(index)] or "unknown" for index in indices],
        dataset.labels[indices],
        [dataset.features[int(index)] for index in indices],
    )


def selected_vectorizer_transform(vectorizer, selected_indices, feature_dicts):
    """Transform directly into the selected feature space when the artifact permits it."""
    indices = np.asarray(selected_indices)
    if (type(vectorizer) is not DictVectorizer or indices.ndim != 1
            or not np.issubdtype(indices.dtype, np.integer)
            or len(np.unique(indices)) != len(indices)
            or np.any(indices < 0) or np.any(indices >= len(vectorizer.feature_names_))):
        return vectorizer.transform(feature_dicts)[:, selected_indices]
    selected = {int(original): position for position, original in enumerate(indices)}
    reduced = copy.copy(vectorizer)
    reduced.vocabulary_ = {name: selected[index] for name, index in vectorizer.vocabulary_.items()
                           if index in selected}
    reduced.feature_names_ = [vectorizer.feature_names_[int(index)] for index in indices]
    return reduced.transform(feature_dicts)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.output)
    prediction_dir = run_dir / "predictions"
    metrics_dir = run_dir / "metrics"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    input_dir = resolve_split_input(Path(args.input), args.split)
    forced_split = args.split if args.split != "all" and input_dir.name == args.split else None
    print("加载数据集中")
    stage_started = perf_counter()
    dataset = load_static_dataset(
        input_dir,
        forced_split=forced_split,
        read_report_split=False if forced_split else True,
        workers=args.workers,
    )
    print(f"数据加载与特征提取耗时: {perf_counter() - stage_started:.2f} 秒")
    print("划分数据集中")
    indices = np.arange(len(dataset.splits)) if forced_split else filter_indices(dataset.splits, args.split)
    if len(indices) == 0:
        raise SystemExit(f"没有找到 split={args.split} 的样本，实际扫描目录: {input_dir}")

    sample_keys, sample_ids, apk_names, splits, y, feature_dicts = subset(dataset, indices)
    print("创建数据集中")
    stage_started = perf_counter()
    model, vectorizer, selected_indices, _ = load_artifacts(Path(args.model_dir))
    print(f"加载模型耗时: {perf_counter() - stage_started:.2f} 秒")
    stage_started = perf_counter()
    X = selected_vectorizer_transform(vectorizer, selected_indices, feature_dicts)
    print(f"特征向量化耗时: {perf_counter() - stage_started:.2f} 秒；矩阵形状: {X.shape}")
    stage_started = perf_counter()
    probs = predict_probability(model, X)
    print(f"模型预测耗时: {perf_counter() - stage_started:.2f} 秒")

    suffix = args.split if args.split != "all" else "all"
    stage_started = perf_counter()
    write_predictions(
        prediction_dir / f"{suffix}_predictions.csv",
        sample_keys,
        sample_ids,
        apk_names,
        splits,
        y,
        probs,
        args.threshold,
    )
    metrics = binary_metrics(y, probs, args.threshold)
    metrics.update({"split": args.split, "n_samples": int(len(y))})
    write_json(metrics_dir / f"{suffix}_metrics.json", metrics)
    print(f"结果写出与指标计算耗时: {perf_counter() - stage_started:.2f} 秒")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
