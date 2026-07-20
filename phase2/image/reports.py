import csv
import json
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd

from phase2.common.identity import build_sample_key


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def classification_report_rows(report: Dict[str, object]):
    rows = []
    for label, values in report.items():
        if isinstance(values, dict):
            row = {"label": label}
            row.update(values)
        else:
            row = {"label": label, "precision": None, "recall": None, "f1-score": values, "support": None}
        rows.append(row)
    return rows


def write_metric_artifacts(output_dir: Path, prefix: str, metrics: Dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = metrics.get("confusion_matrix")
    if matrix is not None:
        pd.DataFrame(
            matrix,
            index=["true_benign", "true_malware"],
            columns=["pred_benign", "pred_malware"],
        ).to_csv(output_dir / f"{prefix}_confusion_matrix.csv")

    report = metrics.get("classification_report")
    if isinstance(report, dict):
        pd.DataFrame(classification_report_rows(report)).to_csv(
            output_dir / f"{prefix}_classification_report.csv",
            index=False,
        )


def write_predictions(
    path: Path,
    sample_infos: Sequence[Dict[str, object]],
    probs: np.ndarray,
    threshold: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_key",
                "sample_id",
                "apk_name",
                "split",
                "y_true",
                "prob_malware",
                "pred",
                "image_path",
            ],
        )
        writer.writeheader()
        for sample, prob in zip(sample_infos, probs):
            label = int(sample["label"])
            split = str(sample["split"])
            sample_id = str(sample["sample_id"])
            writer.writerow(
                {
                    "sample_key": build_sample_key(sample_id, split, label),
                    "sample_id": sample_id,
                    "apk_name": sample.get("apk_name") or f"{sample_id}.apk",
                    "split": split,
                    "y_true": label,
                    "prob_malware": float(prob),
                    "pred": int(float(prob) >= threshold),
                    "image_path": sample.get("image_path", ""),
                }
            )
