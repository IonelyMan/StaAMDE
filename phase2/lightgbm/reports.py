import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_predictions(
    path: Path,
    sample_keys: Sequence[str],
    sample_ids: Sequence[str],
    apk_names: Sequence[str],
    splits: Sequence[str],
    y: np.ndarray,
    probs: np.ndarray,
    threshold: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_key", "sample_id", "apk_name", "split", "y_true", "prob_malware", "pred"],
        )
        writer.writeheader()
        for index, prob in enumerate(probs):
            writer.writerow(
                {
                    "sample_key": sample_keys[index],
                    "sample_id": sample_ids[index],
                    "apk_name": apk_names[index],
                    "split": splits[index],
                    "y_true": int(y[index]),
                    "prob_malware": float(prob),
                    "pred": int(prob >= threshold),
                }
            )


def write_feature_scores(path: Path, feature_names: List[str], mi_scores: np.ndarray, selected: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected_set = set(int(i) for i in selected.tolist())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["feature", "mi_score", "selected"])
        writer.writeheader()
        for index in np.argsort(mi_scores)[::-1]:
            writer.writerow(
                {
                    "feature": feature_names[int(index)],
                    "mi_score": float(mi_scores[int(index)]),
                    "selected": int(int(index) in selected_set),
                }
            )


def write_lightgbm_importance(path: Path, model, selected_names: List[str]) -> None:
    if not hasattr(model, "booster_"):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    gains = model.booster_.feature_importance(importance_type="gain")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["feature", "gain"])
        writer.writeheader()
        for index in np.argsort(gains)[::-1]:
            writer.writerow({"feature": selected_names[int(index)], "gain": float(gains[int(index)])})
