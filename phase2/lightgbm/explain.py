import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from tqdm import tqdm
import numpy as np
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase2.lightgbm.dataset import load_static_dataset
from phase2.lightgbm.inference import filter_indices
from phase2.lightgbm.model import load_artifacts, predict_probability


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="为 LightGBM 静态特征模型生成 SHAP 解释")
    parser.add_argument("--input", required=True, help="phase1/static_feature_extract.py 输出目录")
    parser.add_argument("--model-dir", required=True, help="训练得到的 models 目录")
    parser.add_argument("--output", required=True, help="输出目录，例如 runs/static_lgbm")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--max-samples", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=12, help="静态 JSON 并行加载进程数；0 表示自动")
    parser.add_argument("--method", choices=["native", "shap"], default="native", help="native 使用 LightGBM 原生 pred_contrib")
    parser.add_argument("--batch-size", type=int, default=2048, help="原生 SHAP 分批计算大小")
    parser.add_argument("--top-k", type=int, default=300, help="summary 输出的特征数量")
    parser.add_argument("--save-values", action="store_true", help="额外保存完整 SHAP values npy；大数据会很占空间")
    parser.add_argument("--no-plots", action="store_true", help="只写 CSV，不生成 SHAP 图")
    parser.add_argument(
        "--include-global-plots",
        action="store_true",
        help="指定 waterfall 样本时也重新生成全局 beeswarm/bar；默认不会重复生成全局图",
    )
    parser.add_argument("--summary-plot-top-k", type=int, default=20, help="beeswarm 图中展示的特征数")
    parser.add_argument("--bar-top-k", type=int, default=20, help="Mean |SHAP| bar plot 图中展示的特征数")
    parser.add_argument("--waterfall-top-k", type=int, default=15, help="waterfall 图中展示的单样本主要贡献特征数")
    parser.add_argument("--waterfall-sample-id", default=None, help="指定要画 waterfall 的 sample_id")
    parser.add_argument("--waterfall-index", type=int, default=None, help="按本次 SHAP 子集的局部下标指定 waterfall 样本")
    parser.add_argument("--figure-formats", nargs="+", default=["png"], help="图片格式，例如 png pdf svg")
    parser.add_argument("--figure-dpi", type=int, default=300)
    return parser.parse_args()


def resolve_split_input(input_dir: Path, split: str) -> Path:
    if split == "all":
        return input_dir
    split_dir = input_dir / split
    return split_dir if split_dir.exists() else input_dir


def normalize_lgb_contrib_with_base(contrib, n_features: int) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(contrib, list):
        contrib = contrib[1] if len(contrib) > 1 else contrib[0]
    if sparse.issparse(contrib):
        contrib = contrib.toarray()
    else:
        contrib = np.asarray(contrib)
        if contrib.ndim == 0 and sparse.issparse(contrib.item()):
            contrib = contrib.item().toarray()

    if contrib.ndim == 1:
        contrib = contrib.reshape(1, -1)
    if contrib.ndim == 3:
        if contrib.shape[1] == n_features + 1:
            class_index = 1 if contrib.shape[2] > 1 else 0
            contrib = contrib[:, :, class_index]
        elif contrib.shape[2] == n_features + 1:
            class_index = 1 if contrib.shape[1] > 1 else 0
            contrib = contrib[:, class_index, :]
        else:
            raise ValueError(f"无法识别 LightGBM pred_contrib 输出形状: {contrib.shape}, n_features={n_features}")
    if contrib.shape[1] == n_features + 1:
        return contrib[:, :-1], contrib[:, -1]
    if contrib.shape[1] % (n_features + 1) == 0:
        n_classes = contrib.shape[1] // (n_features + 1)
        contrib = contrib.reshape(contrib.shape[0], n_classes, n_features + 1)
        class_index = 1 if n_classes > 1 else 0
        selected = contrib[:, class_index, :]
        return selected[:, :-1], selected[:, -1]
    if contrib.shape[1] == n_features:
        return contrib, np.zeros(contrib.shape[0], dtype=np.float64)
    raise ValueError(f"无法识别 LightGBM pred_contrib 输出形状: {contrib.shape}, n_features={n_features}")


def normalize_lgb_contrib(contrib, n_features: int) -> np.ndarray:
    values, _ = normalize_lgb_contrib_with_base(contrib, n_features)
    return values


def compute_native_lgb_shap(model, X, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    if not hasattr(model, "booster_"):
        raise ValueError("当前模型没有 booster_，无法使用 LightGBM 原生 SHAP")
    booster = model.booster_
    values = []
    base_values = []
    n_features = X.shape[1]
    for start in tqdm(range(0, X.shape[0], batch_size),desc="正在计算lgb_shap"):
        end = min(start + batch_size, X.shape[0])
        contrib = booster.predict(X[start:end], pred_contrib=True)
        batch_values, batch_base_values = normalize_lgb_contrib_with_base(contrib, n_features)
        values.append(batch_values)
        base_values.append(batch_base_values)
    return np.vstack(values), np.concatenate(base_values)


def compute_shap_package_values(model, X) -> Tuple[np.ndarray, np.ndarray]:
    try:
        import shap
    except ImportError as exc:
        raise SystemExit("缺少 shap，请先安装 requirements.txt 中的依赖，或使用默认 --method native。") from exc

    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X)
    expected_value = explainer.expected_value
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]
    if isinstance(expected_value, (list, tuple, np.ndarray)):
        expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]
    base_values = np.full(values.shape[0], float(np.asarray(expected_value).reshape(-1)[0]), dtype=np.float64)
    return values, base_values


def safe_filename(text: str) -> str:
    keep = []
    for char in str(text):
        keep.append(char if char.isalnum() or char in {"-", "_", "."} else "_")
    return "".join(keep).strip("_") or "sample"


def short_feature_name(name: str, max_len: int = 82) -> str:
    if len(name) <= max_len:
        return name
    head = max_len // 2 - 2
    tail = max_len - head - 5
    return f"{name[:head]} ... {name[-tail:]}"


def save_figure(fig, output_base: Path, formats: Sequence[str], dpi: int) -> List[Path]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        clean_fmt = fmt.lower().lstrip(".")
        path = output_base.with_suffix(f".{clean_fmt}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        paths.append(path)
    return paths


def feature_values_subset(X, feature_indices: Sequence[int]) -> np.ndarray:
    subset = X[:, list(feature_indices)]
    if sparse.issparse(subset):
        subset = subset.toarray()
    return np.asarray(subset, dtype=np.float64)


def scaled_feature_values(values: np.ndarray) -> np.ndarray:
    scaled = np.zeros_like(values, dtype=np.float64)
    for column in range(values.shape[1]):
        col = values[:, column]
        finite = col[np.isfinite(col)]
        if finite.size == 0:
            continue
        low, high = np.percentile(finite, [5, 95])
        if math.isclose(float(low), float(high)):
            scaled[:, column] = 0.5
        else:
            scaled[:, column] = np.clip((col - low) / (high - low), 0.0, 1.0)
    return scaled


def beeswarm_offsets(x: np.ndarray, max_jitter: float = 0.36, bins: int = 40) -> np.ndarray:
    offsets = np.zeros(len(x), dtype=np.float64)
    finite_mask = np.isfinite(x)
    if finite_mask.sum() <= 1:
        return offsets
    finite = x[finite_mask]
    low, high = np.percentile(finite, [1, 99])
    if math.isclose(float(low), float(high)):
        return offsets
    edges = np.linspace(low, high, bins + 1)
    bin_ids = np.digitize(np.clip(x, low, high), edges, right=False)
    for bin_id in np.unique(bin_ids):
        members = np.where(bin_ids == bin_id)[0]
        if len(members) <= 1:
            continue
        step = min(0.085, max_jitter / max(1, math.ceil(len(members) / 2)))
        pattern = [0.0]
        for rank in range(1, len(members)):
            magnitude = (rank + 1) // 2
            sign = 1.0 if rank % 2 else -1.0
            pattern.append(sign * magnitude * step)
        offsets[members] = np.asarray(pattern[: len(members)])
    return np.clip(offsets, -max_jitter, max_jitter)


def plot_manual_beeswarm(
    shap_values: np.ndarray,
    feature_values: np.ndarray,
    feature_names: Sequence[str],
    output_base: Path,
    formats: Sequence[str],
    dpi: int,
) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = np.argsort(np.abs(shap_values).mean(axis=0))
    ordered_values = shap_values[:, order]
    ordered_feature_values = scaled_feature_values(feature_values[:, order])
    ordered_names = [feature_names[int(index)] for index in order]

    fig_height = max(5.5, 0.38 * len(order) + 1.8)
    fig, ax = plt.subplots(figsize=(9.5, fig_height))
    scatter = None
    for y_pos, column in enumerate(range(ordered_values.shape[1])):
        x = ordered_values[:, column]
        y = y_pos + beeswarm_offsets(x)
        scatter = ax.scatter(
            x,
            y,
            c=ordered_feature_values[:, column],
            cmap="coolwarm",
            vmin=0.0,
            vmax=1.0,
            s=12,
            alpha=0.82,
            linewidths=0,
        )
    ax.axvline(0.0, color="#666666", linewidth=0.8)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([short_feature_name(name) for name in ordered_names])
    ax.set_xlabel("SHAP value (impact on malware log-odds)")
    ax.set_title("SHAP Summary Plot")
    ax.grid(axis="x", color="#dddddd", linewidth=0.6, alpha=0.8)
    if scatter is not None:
        colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
        colorbar.set_label("Feature value")
        colorbar.set_ticks([0, 1])
        colorbar.set_ticklabels(["Low", "High"])
    paths = save_figure(fig, output_base, formats, dpi)
    plt.close(fig)
    return paths


def plot_shap_beeswarm(
    shap_values: np.ndarray,
    feature_values: np.ndarray,
    feature_names: Sequence[str],
    output_base: Path,
    formats: Sequence[str],
    dpi: int,
) -> List[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap

        plt.figure()
        shap.summary_plot(
            shap_values,
            feature_values,
            feature_names=list(feature_names),
            max_display=len(feature_names),
            show=False,
            plot_type="dot",
        )
        fig = plt.gcf()
        paths = save_figure(fig, output_base, formats, dpi)
        plt.close(fig)
        return paths
    except Exception as exc:
        print(f"[warning] shap.summary_plot failed, fallback to manual beeswarm: {exc}", file=sys.stderr)
        return plot_manual_beeswarm(shap_values, feature_values, feature_names, output_base, formats, dpi)


def plot_mean_abs_bar(
    mean_abs: np.ndarray,
    feature_names: Sequence[str],
    top_k: int,
    output_base: Path,
    formats: Sequence[str],
    dpi: int,
) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top_indices = np.argsort(mean_abs)[::-1][: max(1, top_k)]
    plot_indices = top_indices[::-1]
    labels = [short_feature_name(feature_names[int(index)]) for index in plot_indices]
    values = mean_abs[plot_indices]

    fig_height = max(5.0, 0.34 * len(plot_indices) + 1.2)
    fig, ax = plt.subplots(figsize=(9.0, fig_height))
    ax.barh(np.arange(len(plot_indices)), values, color="#d94841", alpha=0.88)
    ax.set_yticks(np.arange(len(plot_indices)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Top-{len(plot_indices)} Static Features by Mean |SHAP|")
    ax.grid(axis="x", color="#dddddd", linewidth=0.6, alpha=0.8)
    paths = save_figure(fig, output_base, formats, dpi)
    plt.close(fig)
    return paths


def choose_waterfall_position(
    sample_ids: Sequence[str],
    labels: np.ndarray,
    probs: np.ndarray,
    sample_id: Optional[str],
    local_index: Optional[int],
) -> int:
    if sample_id:
        requested = {str(sample_id), Path(str(sample_id)).stem}
        for pos, current in enumerate(sample_ids):
            if str(current) in requested:
                return pos
        raise SystemExit(f"waterfall sample_id not found in current SHAP subset: {sample_id}")
    if local_index is not None:
        if local_index < 0 or local_index >= len(sample_ids):
            raise SystemExit(f"waterfall-index out of range: {local_index}, n_samples={len(sample_ids)}")
        return int(local_index)

    preds = (probs >= 0.5).astype(np.int64)
    true_positive = np.where((labels == 1) & (preds == 1))[0]
    if len(true_positive) > 0:
        return int(true_positive[np.argmax(probs[true_positive])])
    malware = np.where(labels == 1)[0]
    if len(malware) > 0:
        return int(malware[np.argmax(probs[malware])])
    return int(np.argmax(np.abs(probs - 0.5)))


def dataset_sample_keys(dataset, dataset_index: int) -> set:
    sample_id = str(dataset.sample_ids[dataset_index])
    apk_name = str(dataset.apk_names[dataset_index])
    keys = {sample_id, apk_name}
    if apk_name:
        keys.add(Path(apk_name).stem)
    sample_key = str(dataset.sample_keys[dataset_index])
    if sample_key:
        keys.add(sample_key)
    return {key for key in keys if key}


def find_dataset_index_by_sample_id(dataset, indices: np.ndarray, sample_id: str) -> int:
    requested = str(sample_id).strip()
    for dataset_index in indices.tolist():
        dataset_index = int(dataset_index)
        if requested in dataset_sample_keys(dataset, dataset_index):
            return dataset_index
    examples = [dataset.sample_ids[int(index)] for index in indices[:20]]
    raise SystemExit(
        f"waterfall sample_id not found in split subset: {sample_id}; "
        f"available examples: {examples}"
    )


def sample_indices(
    indices: np.ndarray,
    max_samples: int,
    seed: int,
    keep_index: Optional[int] = None,
) -> np.ndarray:
    if len(indices) <= max_samples:
        return indices
    rng = random.Random(seed)
    sampled = rng.sample(indices.tolist(), max_samples)
    if keep_index is not None and int(keep_index) not in sampled:
        sampled[-1] = int(keep_index)
    return np.asarray(sampled, dtype=np.int64)


def waterfall_rows(
    shap_row: np.ndarray,
    feature_names: Sequence[str],
    feature_values: np.ndarray,
    top_k: int,
) -> List[Dict[str, object]]:
    top_indices = np.argsort(np.abs(shap_row))[::-1][: max(1, top_k)]
    rows: List[Dict[str, object]] = []
    for rank, feature_index in enumerate(top_indices, start=1):
        rows.append(
            {
                "rank": rank,
                "feature": feature_names[int(feature_index)],
                "feature_value": float(feature_values[int(feature_index)]),
                "shap_value": float(shap_row[int(feature_index)]),
                "abs_shap_value": float(abs(shap_row[int(feature_index)])),
            }
        )
    other_value = float(shap_row.sum() - sum(float(row["shap_value"]) for row in rows))
    if not math.isclose(other_value, 0.0, abs_tol=1e-12):
        rows.append(
            {
                "rank": len(rows) + 1,
                "feature": f"other_{len(shap_row) - len(top_indices)}_features",
                "feature_value": "",
                "shap_value": other_value,
                "abs_shap_value": abs(other_value),
            }
        )
    return rows


def plot_waterfall(
    rows: Sequence[Dict[str, object]],
    base_value: float,
    sample_id: str,
    prob: float,
    y_true: int,
    pred: int,
    output_base: Path,
    formats: Sequence[str],
    dpi: int,
) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    display_rows = list(rows)
    running = base_value
    starts = []
    ends = []
    for row in rows:
        starts.append(running)
        running += float(row["shap_value"])
        ends.append(running)

    fig_height = max(5.2, 0.42 * len(display_rows) + 1.8)
    fig, ax = plt.subplots(figsize=(10.0, fig_height))
    y_positions = np.arange(len(display_rows))
    for y_pos, row, start, end in zip(y_positions, display_rows, starts, ends):
        value = float(row["shap_value"])
        color = "#d62728" if value >= 0 else "#1f77b4"
        left = min(start, end)
        width = abs(end - start)
        ax.barh(y_pos, width, left=left, height=0.62, color=color, alpha=0.88)
        ax.text(
            end,
            y_pos,
            f" {value:+.4g}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=8,
            color=color,
        )

    final_value = base_value + sum(float(row["shap_value"]) for row in rows)
    ax.axvline(base_value, color="#666666", linestyle="--", linewidth=1.0, label=f"E[f(x)]={base_value:.4g}")
    ax.axvline(final_value, color="#111111", linewidth=1.1, label=f"f(x)={final_value:.4g}")
    ax.set_yticks(y_positions)
    ax.set_yticklabels([short_feature_name(str(row["feature"])) for row in display_rows])
    ax.invert_yaxis()
    ax.set_xlabel("Model output for malware class (log-odds)")
    ax.set_title(f"SHAP Waterfall: {sample_id} | p_malware={prob:.4f}, y={y_true}, pred={pred}")
    ax.grid(axis="x", color="#dddddd", linewidth=0.6, alpha=0.8)
    ax.legend(loc="best", fontsize=8)
    paths = save_figure(fig, output_base, formats, dpi)
    plt.close(fig)
    return paths


def write_rows(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    reports_dir = Path(args.output) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    input_dir = resolve_split_input(Path(args.input), args.split)
    forced_split: Optional[str] = args.split if args.split != "all" and input_dir.name == args.split else None
    dataset = load_static_dataset(
        input_dir,
        forced_split=forced_split,
        read_report_split=False if forced_split else True,
        workers=args.workers,
    )
    indices = np.arange(len(dataset.splits)) if forced_split else filter_indices(dataset.splits, args.split)
    if len(indices) == 0:
        raise SystemExit(f"没有找到 split={args.split} 的样本，实际扫描目录: {input_dir}")

    specific_waterfall_requested = args.waterfall_sample_id is not None or args.waterfall_index is not None
    generate_global_outputs = (not specific_waterfall_requested) or args.include_global_plots
    keep_dataset_index = None
    if args.waterfall_sample_id:
        keep_dataset_index = find_dataset_index_by_sample_id(dataset, indices, args.waterfall_sample_id)

    if generate_global_outputs:
        indices = sample_indices(indices, args.max_samples, args.seed, keep_index=keep_dataset_index)
    elif args.waterfall_sample_id:
        indices = np.asarray([keep_dataset_index], dtype=np.int64)
    else:
        indices = sample_indices(indices, args.max_samples, args.seed)
        if args.waterfall_index is None:
            raise SystemExit("internal error: local-only mode requires --waterfall-sample-id or --waterfall-index")
        if args.waterfall_index < 0 or args.waterfall_index >= len(indices):
            raise SystemExit(f"waterfall-index out of range: {args.waterfall_index}, n_samples={len(indices)}")
        indices = np.asarray([int(indices[args.waterfall_index])], dtype=np.int64)

    model, vectorizer, selected_indices, selected_names = load_artifacts(Path(args.model_dir))
    feature_dicts = [dataset.features[int(index)] for index in indices]
    X = vectorizer.transform(feature_dicts)[:, selected_indices]
    y = dataset.labels[indices]
    sample_ids = [dataset.sample_ids[int(index)] for index in indices]
    apk_names = [dataset.apk_names[int(index)] for index in indices]
    sample_keys = [dataset.sample_keys[int(index)] for index in indices]
    probs = predict_probability(model, X)

    if args.method == "native":
        values, base_values = compute_native_lgb_shap(model, X, args.batch_size)
    else:
        X_dense = X.toarray() if hasattr(X, "toarray") else X
        values, base_values = compute_shap_package_values(model, X_dense)

    suffix = args.split if args.split != "all" else "all"
    mean_abs = None
    summary_csv = None
    if generate_global_outputs:
        mean_abs = np.abs(values).mean(axis=0)
        summary_rows = [
            {
                "rank": rank,
                "feature": selected_names[int(index)],
                "mean_abs_shap": float(mean_abs[int(index)]),
                "mean_shap": float(values[:, int(index)].mean()),
            }
            for rank, index in enumerate(np.argsort(mean_abs)[::-1][: args.top_k], start=1)
        ]
        summary_csv = reports_dir / f"{suffix}_shap_summary.csv"
        write_rows(
            summary_csv,
            ["rank", "feature", "mean_abs_shap", "mean_shap"],
            summary_rows,
        )
    if args.save_values:
        np.save(reports_dir / f"{suffix}_shap_values.npy", values)

    figure_outputs: Dict[str, List[str]] = {}
    waterfall_sample_id = None
    waterfall_position = None
    if not args.no_plots:
        figures_dir = reports_dir / "shap_figures"
        if generate_global_outputs and mean_abs is not None:
            summary_top_indices = np.argsort(mean_abs)[::-1][: max(1, args.summary_plot_top_k)]
            summary_feature_values = feature_values_subset(X, summary_top_indices)
            summary_shap_values = values[:, summary_top_indices]
            summary_feature_names = [selected_names[int(index)] for index in summary_top_indices]
            figure_outputs["summary_beeswarm"] = [
                str(path)
                for path in plot_shap_beeswarm(
                    summary_shap_values,
                    summary_feature_values,
                    summary_feature_names,
                    figures_dir / f"{suffix}_shap_summary_beeswarm",
                    args.figure_formats,
                    args.figure_dpi,
                )
            ]
            figure_outputs["top_mean_abs_bar"] = [
                str(path)
                for path in plot_mean_abs_bar(
                    mean_abs,
                    selected_names,
                    args.bar_top_k,
                    figures_dir / f"{suffix}_top{args.bar_top_k}_mean_abs_shap_bar",
                    args.figure_formats,
                    args.figure_dpi,
                )
            ]

        waterfall_sample_id_arg = args.waterfall_sample_id
        waterfall_index_arg = args.waterfall_index
        if specific_waterfall_requested and not generate_global_outputs:
            waterfall_sample_id_arg = None
            waterfall_index_arg = 0

        waterfall_position = choose_waterfall_position(
            sample_ids,
            y,
            probs,
            waterfall_sample_id_arg,
            waterfall_index_arg,
        )
        waterfall_sample_id = sample_ids[waterfall_position]
        waterfall_x = X[waterfall_position]
        if sparse.issparse(waterfall_x):
            waterfall_feature_values = np.asarray(waterfall_x.toarray()).reshape(-1)
        else:
            waterfall_feature_values = np.asarray(waterfall_x).reshape(-1)
        local_rows = waterfall_rows(
            values[waterfall_position],
            selected_names,
            waterfall_feature_values,
            args.waterfall_top_k,
        )
        for row in local_rows:
            row.update(
                {
                    "sample_key": sample_keys[waterfall_position],
                    "sample_id": sample_ids[waterfall_position],
                    "apk_name": apk_names[waterfall_position],
                    "y_true": int(y[waterfall_position]),
                    "prob_malware": float(probs[waterfall_position]),
                    "pred": int(probs[waterfall_position] >= 0.5),
                    "base_value": float(base_values[waterfall_position]),
                    "model_output": float(base_values[waterfall_position] + values[waterfall_position].sum()),
                }
            )
        write_rows(
            reports_dir / f"{suffix}_{safe_filename(waterfall_sample_id)}_waterfall_contributions.csv",
            [
                "sample_key",
                "sample_id",
                "apk_name",
                "y_true",
                "prob_malware",
                "pred",
                "base_value",
                "model_output",
                "rank",
                "feature",
                "feature_value",
                "shap_value",
                "abs_shap_value",
            ],
            local_rows,
        )
        figure_outputs["waterfall"] = [
            str(path)
            for path in plot_waterfall(
                local_rows,
                float(base_values[waterfall_position]),
                waterfall_sample_id,
                float(probs[waterfall_position]),
                int(y[waterfall_position]),
                int(probs[waterfall_position] >= 0.5),
                figures_dir / f"{suffix}_{safe_filename(waterfall_sample_id)}_waterfall",
                args.figure_formats,
                args.figure_dpi,
            )
        ]

    config = {
        "input": str(args.input),
        "model_dir": str(args.model_dir),
        "split": args.split,
        "method": args.method,
        "n_samples": int(len(indices)),
        "max_samples": args.max_samples,
        "summary_csv": str(summary_csv) if summary_csv else None,
        "global_outputs_generated": bool(generate_global_outputs),
        "plots_enabled": not args.no_plots,
        "figure_outputs": figure_outputs,
        "waterfall_sample_id": waterfall_sample_id,
        "waterfall_position": int(waterfall_position) if waterfall_position is not None else None,
        "note": "Native LightGBM SHAP values explain the malware-class raw score/log-odds; positive values push the sample toward malware.",
    }
    (reports_dir / f"{suffix}_shap_figure_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if summary_csv:
        print(f"SHAP summary written: {summary_csv}")
    elif specific_waterfall_requested:
        print("SHAP global summary/beeswarm/bar skipped for sample-specific waterfall mode.")
    if figure_outputs:
        print(json.dumps(figure_outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


"""
现在运行 explain.py 默认会生成：
runs/static_lgbm/reports/test_shap_summary.csv

# 每个featrue在横轴有很多点，往左点数越多表示这个featrue更让模型觉得是良性，往右则是恶意
runs/static_lgbm/reports/shap_figures/test_shap_summary_beeswarm.png/pdf

# 模型整体最依赖哪些特征，全局 Feature Importance；柱子越长说明这个Feature越重要。
runs/static_lgbm/reports/shap_figures/test_top20_mean_abs_shap_bar.png/pdf  

# 解释为什么这个APK会被判成恶意，不同APK有不同的TOP-K特征。同一个Feature对于不同APK会有时是正，有时是负
runs/static_lgbm/reports/shap_figures/test_<sample_id>_waterfall.png/pdf
runs/static_lgbm/reports/test_<sample_id>_waterfall_contributions.csv
runs/static_lgbm/reports/test_shap_figure_config.json

推荐命令：
python -m phase2.lightgbm.explain \
  --input /home/linux/7T/lzw/datasets/android_zoo/android_static \
  --model-dir runs/lightgbm/07-04_12-58/models \
  --output explain/ligthgbm \
  --split test \
  --workers 4 \
  --max-samples 800 \
  --summary-plot-top-k 20 \
  --bar-top-k 20 \
  --waterfall-top-k 15

大约执行三分钟
如果要指定 waterfall 的 APK：
python -m phase2.lightgbm.explain \
  --input /home/linux/7T/lzw/datasets/android_zoo/android_static \
  --model-dir runs/lightgbm/07-04_12-58/models \
  --output explain/lightgbm \
  --split test \
  --waterfall-sample-id com.thecybernanny.adroapp

它会默认只生成这个样本的：
runs/static_lgbm/reports/test_com.thecybernanny.adroapp_waterfall_contributions.csv
runs/static_lgbm/reports/shap_figures/test_com.thecybernanny.adroapp_waterfall.png
不会再重复生成：
test_shap_summary_beeswarm.png
test_top20_mean_abs_shap_bar.png
而且不只是跳过画图，我还改了计算逻辑：指定 --waterfall-sample-id 时默认只对这个 APK 计算 SHAP，不再对 --max-samples 800 里的所有样本算一遍，所以速度会明显快。

如果你确实想在指定样本时也重新生成全局图，再加：
--include-global-plots

实现细节也补好了：
beeswarm：优先用标准 shap.summary_plot；如果环境没有 shap 绘图能力，会自动退回内置 beeswarm-style，不会中断。
bar plot：Top-20 Mean |SHAP|，方便论文里做全局特征重要性对比。
waterfall：保留 LightGBM 原生 pred_contrib 的 base value，能从 E[f(x)] 推到单样本 f(x)，并输出贡献明细 CSV。
默认 waterfall 样本选择：真实恶意、预测恶意、恶意概率最高的样本。
"""