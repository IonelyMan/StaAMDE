import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from phase2.common.splits import VALID_SPLITS
from phase2.common.identity import build_sample_key
from phase2.lightgbm.features import flatten_report


@dataclass
class StaticDataset:
    sample_keys: List[str]
    sample_ids: List[str]
    apk_names: List[str]
    labels: np.ndarray
    splits: List[Optional[str]]
    features: List[Dict[str, float]]


def iter_reports(input_dir: Path) -> Iterable[Path]:
    yield from sorted(input_dir.rglob("static_features_report.json"))


def infer_split(
    report: Dict[str, object],
    report_path: Path,
    input_dir: Path,
    forced_split: Optional[str] = None,
    read_report_split: bool = True,
) -> Optional[str]:
    if forced_split:
        return forced_split
    if read_report_split:
        value = report.get("split") or report.get("belong")
        if isinstance(value, str) and value.lower() in VALID_SPLITS:
            return value.lower()
    if input_dir.name.lower() in VALID_SPLITS:
        return input_dir.name.lower()
    try:
        parts = [part.lower() for part in report_path.relative_to(input_dir).parts]
    except ValueError:
        parts = [part.lower() for part in report_path.parts]
    for part in parts:
        if part in VALID_SPLITS:
            return part
    return None


def infer_label(report: Dict[str, object], report_path: Path) -> int:
    label = report.get("label")
    if label is not None:
        return int(label)
    for parent in report_path.parents:
        name = parent.name.lower()
        if name in {"mal", "malware", "1"}:
            return 1
        if name in {"benign", "leg", "normal", "0"}:
            return 0
    raise ValueError(f"无法从报告或路径推断标签: {report_path}")


def load_one_report(
    args: Tuple[str, str, Optional[str], bool],
) -> Optional[Tuple[str, str, str, int, Optional[str], Dict[str, float]]]:
    report_text, input_text, forced_split, read_report_split = args
    report_path = Path(report_text)
    input_dir = Path(input_text)
    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if "error" in report:
        return None

    apk_name = str(report.get("apk_name") or f"{report_path.parent.name}.apk")
    sample_id = Path(apk_name).stem
    split = infer_split(
        report,
        report_path,
        input_dir,
        forced_split=forced_split,
        read_report_split=read_report_split,
    )
    label = infer_label(report, report_path)
    return sample_id, apk_name, build_sample_key(sample_id, split, label), label, split, flatten_report(report)


def resolve_workers(workers: Optional[int]) -> int:
    if workers is None:
        return 1
    if workers <= 0:
        return max(1, min(os.cpu_count() or 1, 8))
    return workers


def load_static_dataset(
    input_dir: Path,
    forced_split: Optional[str] = None,
    read_report_split: bool = True,
    workers: Optional[int] = 1,
) -> StaticDataset:
    sample_ids: List[str] = []
    sample_keys: List[str] = []
    apk_names: List[str] = []
    labels: List[int] = []
    splits: List[Optional[str]] = []
    features: List[Dict[str, float]] = []
    report_paths = list(iter_reports(input_dir))
    worker_count = resolve_workers(workers)

    tasks = [(str(path), str(input_dir), forced_split, read_report_split) for path in report_paths]
    if worker_count <= 1 or len(tasks) <= 1:
        rows = [load_one_report(task) for task in tasks]
    else:
        chunksize = max(1, len(tasks) // (worker_count * 8))
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            rows = list(executor.map(load_one_report, tasks, chunksize=chunksize))

    for row in rows:
        if row is None:
            continue
        sample_id, apk_name, sample_key, label, split, feature = row
        sample_ids.append(sample_id)
        sample_keys.append(sample_key)
        apk_names.append(apk_name)
        labels.append(label)
        splits.append(split)
        features.append(feature)

    if not features:
        raise ValueError(f"没有在 {input_dir} 下找到有效 static_features_report.json")

    return StaticDataset(
        sample_keys=sample_keys,
        sample_ids=sample_ids,
        apk_names=apk_names,
        labels=np.asarray(labels, dtype=np.int64),
        splits=splits,
        features=features,
    )


def concat_static_datasets(datasets: List[StaticDataset]) -> StaticDataset:
    return StaticDataset(
        sample_keys=[item for dataset in datasets for item in dataset.sample_keys],
        sample_ids=[item for dataset in datasets for item in dataset.sample_ids],
        apk_names=[item for dataset in datasets for item in dataset.apk_names],
        labels=np.concatenate([dataset.labels for dataset in datasets]),
        splits=[item for dataset in datasets for item in dataset.splits],
        features=[item for dataset in datasets for item in dataset.features],
    )


def load_static_train_val_dataset(input_dir: Path, workers: Optional[int] = 1) -> StaticDataset:
    train_dir = input_dir / "train"
    val_dir = input_dir / "val"
    if train_dir.exists() and val_dir.exists():
        return concat_static_datasets(
            [
                load_static_dataset(train_dir, forced_split="train", read_report_split=False, workers=workers),
                load_static_dataset(val_dir, forced_split="val", read_report_split=False, workers=workers),
            ]
        )
    return load_static_dataset(input_dir, workers=workers)
