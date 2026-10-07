"""Write LightGBM SHAP evidence and figures for selected APKs or a global cohort."""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase2.lightgbm.dataset import iter_reports, load_static_dataset
from phase2.lightgbm.inference import filter_indices
from phase2.lightgbm.model import load_artifacts, predict_probability
from phase2.lightgbm.shap_plots import plot_global_bar, plot_global_summary, plot_local_waterfall


def parse_args():
    parser = argparse.ArgumentParser(description="Export LightGBM SHAP evidence as CSV and figures")
    parser.add_argument("--input", required=True, help="静态特征报告目录")
    parser.add_argument("--model-dir", required=True, help="LightGBM models 目录")
    parser.add_argument("--output", default="ml_explain", help="解释结果根目录")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--sample-id", nargs="+", action="append", default=[],
                      help="指定一个或多个 APK；可重复传入，也可用逗号分隔")
    mode.add_argument("--global", dest="global_explain", action="store_true", help="计算样本集合的全局 SHAP 摘要")
    parser.add_argument("--max-samples", type=int, default=800, help="全局模式的抽样上限")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--top-k", type=int, default=100, help="局部每个方向或全局保存的特征数")
    parser.add_argument("--waterfall-top-k", type=int, default=15, help="单样本瀑布图展示的主要特征数")
    parser.add_argument("--global-plot-top-k", type=int, default=20, help="全局图展示的特征数")
    parser.add_argument("--figure-format", choices=["png", "pdf", "svg"], default="png")
    parser.add_argument("--figure-dpi", type=int, default=220)
    return parser.parse_args()


def safe_name(value: str) -> str:
    name = Path(str(value)).name
    if name.lower().endswith(".apk"):
        name = name[:-4]
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in name).strip("._") or "sample"


def write_rows(path: Path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def normalize_contributions(contributions, feature_count):
    """Return malware-class feature contributions and the LightGBM base value."""
    if isinstance(contributions, list):
        contributions = contributions[1] if len(contributions) > 1 else contributions[0]
    if sparse.issparse(contributions):
        contributions = contributions.toarray()
    contributions = np.asarray(contributions)
    if contributions.ndim == 1:
        contributions = contributions.reshape(1, -1)
    if contributions.ndim == 3:
        if contributions.shape[1] == feature_count + 1:
            contributions = contributions[:, :, 1 if contributions.shape[2] > 1 else 0]
        elif contributions.shape[2] == feature_count + 1:
            contributions = contributions[:, 1 if contributions.shape[1] > 1 else 0, :]
    if contributions.shape[1] == feature_count + 1:
        return contributions[:, :-1], contributions[:, -1]
    if contributions.shape[1] % (feature_count + 1) == 0:
        classes = contributions.shape[1] // (feature_count + 1)
        selected = contributions.reshape(-1, classes, feature_count + 1)[:, 1 if classes > 1 else 0, :]
        return selected[:, :-1], selected[:, -1]
    raise ValueError(f"Unexpected LightGBM pred_contrib shape: {contributions.shape}")


def compute_contributions(model, matrix, batch_size):
    values, bases = [], []
    for start in range(0, matrix.shape[0], batch_size):
        contribution = model.booster_.predict(matrix[start:start + batch_size], pred_contrib=True)
        batch_values, batch_bases = normalize_contributions(contribution, matrix.shape[1])
        values.append(batch_values)
        bases.append(batch_bases)
    return np.vstack(values), np.concatenate(bases)


def sample_keys(dataset, index):
    apk_name = str(dataset.apk_names[index])
    return {str(dataset.sample_ids[index]), apk_name, Path(apk_name).stem, str(dataset.sample_keys[index])}


def normalize_requested_sample_ids(groups):
    requested = []
    seen = set()
    for group in groups:
        for value in group if isinstance(group, (list, tuple)) else [group]:
            for item in str(value).split(","):
                item = item.strip()
                if item and item not in seen:
                    requested.append(item)
                    seen.add(item)
    return requested


def sample_stem(requested: str) -> str:
    parts = requested.split(":", 2)
    if len(parts) == 3 and parts[0] in {"train", "val", "test"} and parts[1] in {"0", "1"}:
        requested = parts[2]
    name = Path(requested).name
    return name[:-4] if name.lower().endswith(".apk") else name


def direct_sample_reports(input_dir: Path, requested: str):
    """Locate phase1 reports by directory name without parsing the whole split."""
    stem = sample_stem(requested)
    if stem in {"", ".", ".."}:
        return []
    roots = [input_dir]
    roots.extend(input_dir / split for split in ("train", "val", "test") if (input_dir / split).is_dir())
    paths = {
        parent / stem / "static_features_report.json"
        for root in roots
        for parent in (root, root / "mal", root / "benign")
    }
    return sorted(path for path in paths if path.is_file())


def load_requested_dataset(input_dir: Path, forced_split, split: str, requested_ids, workers: int):
    """Read requested APKs by path, falling back for older directory layouts."""
    options = {"forced_split": forced_split, "read_report_split": not bool(forced_split)}
    direct_paths = sorted({path for requested in requested_ids for path in direct_sample_reports(input_dir, requested)})

    def contains_every_requested(dataset):
        indices = np.arange(len(dataset.splits)) if forced_split else filter_indices(dataset.splits, split)
        return all(any(requested in sample_keys(dataset, int(index)) for index in indices)
                   for requested in requested_ids)

    if direct_paths:
        dataset = load_static_dataset(input_dir, workers=1, report_paths=direct_paths, **options)
        if contains_every_requested(dataset):
            return dataset

    # Nonstandard directory layouts still avoid parsing unrelated report contents.
    stems = {sample_stem(requested) for requested in requested_ids}
    named_paths = [path for path in iter_reports(input_dir) if path.parent.name in stems]
    if named_paths and named_paths != direct_paths:
        dataset = load_static_dataset(input_dir, workers=1, report_paths=named_paths, **options)
        if contains_every_requested(dataset):
            return dataset

    # Reports whose apk_name differs from their parent directory require the original scan.
    return load_static_dataset(input_dir, workers=workers, **options)


def main():
    args = parse_args()
    requested_ids = normalize_requested_sample_ids(args.sample_id)
    if not args.global_explain and not requested_ids:
        raise SystemExit("--sample-id requires at least one nonempty APK identifier")
    if min(args.batch_size, args.top_k, args.max_samples, args.waterfall_top_k,
           args.global_plot_top_k, args.figure_dpi) < 1:
        raise SystemExit("Sample limits, plot limits, batch size and figure DPI must be positive")

    input_root = Path(args.input)
    split_dir = input_root / args.split
    input_dir = split_dir if args.split != "all" and split_dir.is_dir() else input_root
    forced_split = args.split if input_dir == split_dir else None
    if requested_ids:
        dataset = load_requested_dataset(input_dir, forced_split, args.split, requested_ids, args.workers)
    else:
        dataset = load_static_dataset(
            input_dir, forced_split=forced_split, read_report_split=not bool(forced_split), workers=args.workers
        )
    indices = np.arange(len(dataset.splits)) if forced_split else filter_indices(dataset.splits, args.split)
    if not len(indices):
        raise SystemExit(f"No samples found for split={args.split} in {input_dir}")

    if requested_ids:
        matches = []
        for requested in requested_ids:
            found = [int(i) for i in indices if requested in sample_keys(dataset, int(i))]
            if len(found) != 1:
                raise SystemExit(f"Expected exactly one APK for --sample-id={requested}; found {len(found)}")
            if found[0] not in matches:
                matches.append(found[0])
        indices = np.asarray(matches, dtype=np.int64)
    elif len(indices) > args.max_samples:
        indices = np.random.default_rng(args.seed).choice(indices, args.max_samples, replace=False)

    model, vectorizer, selected_indices, feature_names = load_artifacts(Path(args.model_dir))
    matrix = vectorizer.transform([dataset.features[int(i)] for i in indices])[:, selected_indices]
    values, bases = compute_contributions(model, matrix, args.batch_size)
    output_root = Path(args.output)

    if args.global_explain:
        mean_abs = np.abs(values).mean(axis=0)
        rows = [
            {"rank": rank, "feature": feature_names[int(i)], "n_samples": int(len(indices)),
             "mean_abs_shap": float(mean_abs[i]),
             "mean_shap": float(values[:, i].mean())}
            for rank, i in enumerate(np.argsort(mean_abs)[::-1][:args.top_k], 1)
        ]
        path = output_root / "global" / "lightgbm" / f"{args.split}_shap_summary.csv"
        write_rows(path, ["rank", "feature", "n_samples", "mean_abs_shap", "mean_shap"], rows)
        bar_path = plot_global_bar(
            mean_abs, feature_names, len(indices),
            path.parent / f"{args.split}_shap_mean_abs_bar.{args.figure_format}",
            args.global_plot_top_k, args.figure_dpi,
        )
        summary_path = plot_global_summary(
            values, matrix, feature_names, mean_abs,
            path.parent / f"{args.split}_shap_summary.{args.figure_format}",
            args.global_plot_top_k, args.figure_dpi,
        )
        print(f"SHAP figures: {bar_path}, {summary_path}")
    else:
        probabilities = predict_probability(model, matrix)
        directory_names = [safe_name(dataset.apk_names[int(index)]) for index in indices]
        if len(directory_names) != len({name.casefold() for name in directory_names}):
            raise SystemExit("Multiple selected APKs share an output directory name; use unique APK names.")
        for position, index_value in enumerate(indices):
            index = int(index_value)
            sample_values = values[position]
            feature_values = (matrix[position].toarray().ravel() if sparse.issparse(matrix)
                              else np.asarray(matrix[position]).ravel())
            probability = float(probabilities[position])
            y_true = int(dataset.labels[index])
            y_pred = int(probability >= 0.5)
            positive = np.flatnonzero(sample_values > 0)
            negative = np.flatnonzero(sample_values < 0)
            selected = list(positive[np.argsort(sample_values[positive])[::-1][:args.top_k]])
            selected += list(negative[np.argsort(sample_values[negative])[:args.top_k]])
            selected.sort(key=lambda i: abs(sample_values[i]), reverse=True)
            rows = [
                {"sample_key": dataset.sample_keys[index], "sample_id": dataset.sample_ids[index],
                 "apk_name": dataset.apk_names[index], "y_true": y_true,
                 "prob_malware": probability, "pred": y_pred,
                 "base_value": float(bases[position]),
                 "model_output": float(bases[position] + sample_values.sum()),
                 "rank": rank, "feature": feature_names[int(i)], "feature_value": float(feature_values[i]),
                 "shap_value": float(sample_values[i]), "abs_shap_value": float(abs(sample_values[i]))}
                for rank, i in enumerate(selected, 1)
            ]
            path = output_root / directory_names[position] / "shap_contributions.csv"
            write_rows(path, ["sample_key", "sample_id", "apk_name", "y_true", "prob_malware", "pred",
                              "base_value", "model_output", "rank", "feature", "feature_value",
                              "shap_value", "abs_shap_value"], rows)
            figure_path = plot_local_waterfall(
                sample_values, feature_names, float(bases[position]), probability, dataset.sample_ids[index],
                path.parent / f"shap_waterfall.{args.figure_format}",
                args.waterfall_top_k, args.figure_dpi,
                y_true=y_true, y_pred=y_pred,
            )
            print(f"SHAP figure: {figure_path}")
            print(f"SHAP evidence: {path}")
    if args.global_explain:
        print(f"SHAP evidence: {path}")


if __name__ == "__main__":
    main()
