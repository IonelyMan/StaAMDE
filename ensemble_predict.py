import argparse
import itertools
import json
from functools import reduce
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score

from phase2.common.identity import build_sample_key
from phase2.common.metrics import binary_metrics


def parse_named_path(spec: str) -> Tuple[str, Path]:
    if "=" not in spec:
        raise ValueError("输入格式应为 name=predictions.csv，例如 static=phase2_runs/static/predictions.csv")
    name, path = spec.split("=", 1)
    name = name.strip()
    if name not in {"static", "graph"}:
        raise ValueError("只支持 static 和 graph 两个模型输入")
    return name, Path(path)


def load_prediction(spec: str, required_split: str = None) -> pd.DataFrame:
    name, path = parse_named_path(spec)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {"prob_malware"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} 必须包含 prob_malware 列")

    if required_split is not None:
        required_metadata = {"sample_id", "split", "y_true"}
        if not required_metadata.issubset(df.columns):
            raise ValueError(f"调参时 {path} 必须包含 sample_id、split、y_true 列")
        if df.empty or df[list(required_metadata)].isna().any().any():
            raise ValueError(f"{path} 没有样本或样本身份、划分、标签存在缺失")
        if not df["split"].eq(required_split).all():
            raise ValueError(f"{path} 必须仅包含 split={required_split} 的样本")
        if not df["y_true"].isin([0, 1]).all():
            raise ValueError(f"{path} 的 y_true 必须全部为 0 或 1")
        if df["sample_id"].astype(str).str.strip().eq("").any() or df["sample_id"].duplicated().any():
            raise ValueError(f"{path} 的 sample_id 不能为空或重复")
        expected_keys = [build_sample_key(str(sid), required_split, int(label))
                         for sid, label in zip(df["sample_id"], df["y_true"])]
        if "sample_key" in df.columns and not df["sample_key"].eq(pd.Series(expected_keys, index=df.index)).all():
            raise ValueError(f"{path} 的 sample_key 与 sample_id、split 或 y_true 不一致")

    if "sample_key" not in df.columns:
        fallback_required = {"sample_id", "split", "y_true"}
        if not fallback_required.issubset(df.columns):
            raise ValueError(
                f"{path} 必须包含 sample_key；或同时包含 sample_id,split,y_true 以便自动构造 sample_key"
            )
        df["sample_key"] = [
            build_sample_key(str(sample_id), str(split), int(label))
            for sample_id, split, label in zip(df["sample_id"], df["split"], df["y_true"])
        ]

    if df["sample_key"].duplicated().any():
        duplicated = df.loc[df["sample_key"].duplicated(), "sample_key"].head(5).tolist()
        raise ValueError(f"{path} 存在重复 sample_key，无法可靠集成: {duplicated}")

    keep = ["sample_key", "prob_malware"]
    optional = [col for col in ["apk_name", "split", "y_true"] if col in df.columns]
    if "sample_id" in df.columns:
        optional.insert(0, "sample_id")
    keep.extend(optional)
    df = df[keep].copy()
    df = df.rename(columns={"prob_malware": f"prob_{name}"})
    for col in optional:
        df = df.rename(columns={col: f"{col}_{name}"})
    return df


def merge_predictions(inputs: List[str], required_split: str = None,
                      source_sample_ids: set = None) -> Tuple[pd.DataFrame, List[str]]:
    names = [parse_named_path(spec)[0] for spec in inputs]
    if len(names) != 2 or set(names) != {"static", "graph"}:
        raise ValueError("必须分别提供一个 static 和一个 graph 预测 CSV")
    frames = [load_prediction(spec, required_split=required_split) for spec in inputs]
    if required_split is not None:
        if source_sample_ids is not None:
            for name, frame in zip(names, frames):
                source_sample_ids.update(frame[f"sample_id_{name}"].astype(str))
        left, right = frames
        paired = pd.merge(left, right, left_on=f"sample_id_{names[0]}",
                          right_on=f"sample_id_{names[1]}", how="inner")
        if not paired[f"y_true_{names[0]}"].eq(paired[f"y_true_{names[1]}"]).all():
            raise ValueError(f"split={required_split} 两个分支存在相同 sample_id 但标签不一致")
    merged = reduce(lambda left, right: pd.merge(left, right, on="sample_key", how="inner"), frames)
    if merged.empty:
        raise ValueError("各模态没有可对齐的 sample_key，请检查 predictions.csv")
    for name, frame in zip(names, frames):
        unmatched = set(frame["sample_key"]) - set(merged["sample_key"])
        if unmatched:
            print(f"[warning] {name} 有 {len(unmatched)} 个样本未能与另一分支对齐，"
                  f"不会进入集成结果；示例: {sorted(unmatched)[:5]}")

    sample_id_cols = [col for col in merged.columns if col.startswith("sample_id_")]
    apk_cols = [col for col in merged.columns if col.startswith("apk_name_")]
    split_cols = [col for col in merged.columns if col.startswith("split_")]
    label_cols = [col for col in merged.columns if col.startswith("y_true_")]
    if required_split is not None:
        for columns in (sample_id_cols, split_cols, label_cols):
            if len(columns) != 2 or not merged[columns[0]].eq(merged[columns[1]]).all():
                raise ValueError(f"两个分支的样本身份、划分或标签不一致: {columns}")
    if sample_id_cols:
        merged["sample_id"] = merged[sample_id_cols].bfill(axis=1).iloc[:, 0]
    if apk_cols:
        merged["apk_name"] = merged[apk_cols].bfill(axis=1).iloc[:, 0]
    if split_cols:
        merged["split"] = merged[split_cols].bfill(axis=1).iloc[:, 0]
    if label_cols:
        merged["y_true"] = merged[label_cols].bfill(axis=1).iloc[:, 0].astype(int)
    return merged, names


def parse_weights(weight_specs: List[str], model_names: List[str]) -> Dict[str, float]:
    if not weight_specs:
        equal = 1.0 / len(model_names)
        return {name: equal for name in model_names}

    weights = {name: 0.0 for name in model_names}
    for spec in weight_specs:
        if "=" not in spec:
            raise ValueError("--weights 格式应为 name=0.5")
        name, value = spec.split("=", 1)
        if name not in weights:
            raise ValueError(f"未知模态权重: {name}")
        weights[name] = float(value)
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("权重总和必须大于 0")
    return {name: value / total for name, value in weights.items()}


def ensemble_probability(df: pd.DataFrame, weights: Dict[str, float]) -> np.ndarray:
    prob = np.zeros(len(df), dtype=np.float64)
    for name, weight in weights.items():
        prob += weight * df[f"prob_{name}"].to_numpy(dtype=np.float64)
    return prob


def metrics_for(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> Dict[str, object]:
    return binary_metrics(y_true, prob, threshold)


def classification_report_rows(report: Dict[str, object]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for label, values in report.items():
        if isinstance(values, dict):
            row = {"label": label}
            row.update(values)
        else:
            row = {"label": label, "precision": None, "recall": None, "f1-score": values, "support": None}
        rows.append(row)
    return rows


def write_metric_artifacts(output_dir: Path, prefix: str, metrics: Dict[str, object]) -> None:
    matrix = metrics.get("confusion_matrix")
    if matrix is not None:
        matrix_df = pd.DataFrame(
            matrix,
            index=["true_benign", "true_malware"],
            columns=["pred_benign", "pred_malware"],
        )
        matrix_df.to_csv(output_dir / f"{prefix}_confusion_matrix.csv")

    report = metrics.get("classification_report")
    if isinstance(report, dict):
        pd.DataFrame(classification_report_rows(report)).to_csv(
            output_dir / f"{prefix}_classification_report.csv",
            index=False,
        )


def tune_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, 181):
        f1 = f1_score(y_true, (prob >= threshold).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_threshold = float(threshold)
    return best_threshold


def pr_auc_for_weights(y_true: np.ndarray, prob: np.ndarray) -> float:
    if len(set(y_true.tolist())) < 2:
        return 0.0
    return float(average_precision_score(y_true, prob))


def generate_weight_grid(names: List[str], step: float):
    ticks = int(round(1.0 / step))
    if len(names) == 1:
        yield {names[0]: 1.0}
        return
    for values in itertools.product(range(ticks + 1), repeat=len(names)):
        if sum(values) != ticks:
            continue
        weights = {name: value / ticks for name, value in zip(names, values)}
        if any(value > 0 for value in weights.values()):
            yield weights


def tune_weights(df: pd.DataFrame, names: List[str], split: str, step: float) -> Dict[str, float]:
    if split != "val" or "split" not in df.columns or "y_true" not in df.columns:
        raise ValueError("权重搜索必须使用带有标签和 split 的验证集")
    target = df[df["split"] == "val"]
    if target.empty:
        raise ValueError("没有验证集样本，不能搜索权重")

    y_true = target["y_true"].to_numpy(dtype=int)
    best_weights = {name: 1.0 / len(names) for name in names}
    best_score = -1.0
    for weights in generate_weight_grid(names, step):
        prob = ensemble_probability(target, weights)
        score = pr_auc_for_weights(y_true, prob)
        if score > best_score:
            best_score = float(score)
            best_weights = weights
    return best_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="静态特征与图模型的恶意概率加权集成")
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="name=predictions.csv，可重复传入，例如 static=... graph=...",
    )
    parser.add_argument("--output", required=True, help="输出目录")
    parser.add_argument("--weights", action="append", default=[], help="name=权重；不传则等权")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tune-threshold", action="store_true", help="在 tune-split 上搜索 F1 最优阈值")
    parser.add_argument("--tune-weights", action="store_true", help="在 tune-split 上粗搜索 PR AUC 最优权重")
    parser.add_argument("--tune-input", action="append", default=[],
                        help="独立验证集预测 CSV，分别传入 static=... 和 graph=...")
    parser.add_argument("--tune-split", choices=["val"], default="val")
    parser.add_argument("--weight-step", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tuning = args.tune_weights or args.tune_threshold
    if tuning and not args.tune_input:
        raise SystemExit("自动搜索必须通过 --tune-input 分别提供两个分支的验证集预测文件")
    if args.tune_input and not tuning:
        raise SystemExit("提供 --tune-input 时必须启用 --tune-weights 或 --tune-threshold")
    output_dir = Path(args.output)

    test_ids = set()
    merged, names = merge_predictions(args.input, required_split="test" if tuning else None,
                                      source_sample_ids=test_ids)
    validation = None
    if tuning:
        validation_ids = set()
        validation, validation_names = merge_predictions(args.tune_input, required_split="val",
                                                         source_sample_ids=validation_ids)
        if set(validation_names) != set(names):
            raise ValueError("验证集和测试集的模型分支不一致")
        overlap = test_ids & validation_ids
        if validation["y_true"].nunique() != 2:
            raise ValueError("验证集必须同时包含良性和恶性样本，才能进行调参")
    weights = parse_weights(args.weights, names)
    if args.tune_weights:
        weights = tune_weights(validation, names, args.tune_split, args.weight_step)

    prob = ensemble_probability(merged, weights)
    threshold = args.threshold
    if args.tune_threshold:
        threshold = tune_threshold(validation["y_true"].to_numpy(dtype=int), ensemble_probability(validation, weights))

    merged["prob_malware"] = prob
    merged["pred"] = (merged["prob_malware"] >= threshold).astype(int)
    output_cols = ["sample_key"]
    if "sample_id" in merged.columns:
        output_cols.append("sample_id")
    if "apk_name" in merged.columns:
        output_cols.append("apk_name")
    if "split" in merged.columns:
        output_cols.append("split")
    if "y_true" in merged.columns:
        output_cols.append("y_true")
    output_cols.extend([f"prob_{name}" for name in names])
    output_cols.extend(["prob_malware", "pred"])
    output_dir.mkdir(parents=True, exist_ok=True)
    merged[output_cols].to_csv(output_dir / "ensemble_predictions.csv", index=False)

    metrics = {
        "weights": weights,
        "threshold": threshold,
        "n_samples": int(len(merged)),
        "models": names,
    }
    if tuning:
        metrics["tuning"] = {"split": "val", "n_samples": int(len(validation)),
                             "inputs": args.tune_input, "tune_weights": args.tune_weights,
                             "tune_threshold": args.tune_threshold}
    if "y_true" in merged.columns:
        y_true = merged["y_true"].to_numpy(dtype=int)
        metrics["all"] = metrics_for(y_true, prob, threshold)
        write_metric_artifacts(output_dir, "ensemble", metrics["all"])
        if "split" in merged.columns:
            for split_name, group in merged.groupby("split"):
                metrics[str(split_name)] = metrics_for(
                    group["y_true"].to_numpy(dtype=int),
                    group["prob_malware"].to_numpy(dtype=float),
                    threshold,
                )
                write_metric_artifacts(output_dir, str(split_name), metrics[str(split_name)])

    metrics_text = json.dumps(metrics, ensure_ascii=False, indent=2)
    (output_dir / "ensemble_metrics.json").write_text(metrics_text, encoding="utf-8")
    (output_dir / "metrics.json").write_text(metrics_text, encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
