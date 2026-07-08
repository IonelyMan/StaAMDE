from collections import Counter
from typing import List, Optional, Sequence, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

VALID_SPLITS = {"train", "val", "test"}


def safe_stratify(y: np.ndarray):
    counts = Counter(y.tolist())
    return y if len(counts) == 2 and min(counts.values()) >= 2 else None


def random_split_indices(
    y: np.ndarray,
    test_size: float,
    val_size: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(y))
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=safe_stratify(y),
    )
    relative_val = val_size / max(1e-9, 1.0 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=relative_val,
        random_state=seed,
        stratify=safe_stratify(y[train_val_idx]),
    )
    return train_idx, val_idx, test_idx


def split_indices_auto(
    y: np.ndarray,
    splits: Sequence[Optional[str]],
    test_size: float,
    val_size: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    normalized = [split.lower() if isinstance(split, str) else None for split in splits]
    split_counts = Counter(split for split in normalized if split)
    if all(split_counts.get(split, 0) > 0 for split in ["train", "val", "test"]):
        train_idx = np.asarray([idx for idx, split in enumerate(normalized) if split == "train"], dtype=np.int64)
        val_idx = np.asarray([idx for idx, split in enumerate(normalized) if split == "val"], dtype=np.int64)
        test_idx = np.asarray([idx for idx, split in enumerate(normalized) if split == "test"], dtype=np.int64)
        return train_idx, val_idx, test_idx, "existing"
    if split_counts.get("train", 0) > 0 and split_counts.get("val", 0) > 0:
        train_idx = np.asarray([idx for idx, split in enumerate(normalized) if split == "train"], dtype=np.int64)
        val_idx = np.asarray([idx for idx, split in enumerate(normalized) if split == "val"], dtype=np.int64)
        test_idx = np.asarray([], dtype=np.int64)
        return train_idx, val_idx, test_idx, "existing_train_val"
    train_idx, val_idx, test_idx = random_split_indices(y, test_size, val_size, seed)
    return train_idx, val_idx, test_idx, "random"


def split_map(train_idx: Sequence[int], val_idx: Sequence[int], test_idx: Sequence[int]) -> dict:
    mapping = {int(index): "train" for index in train_idx}
    mapping.update({int(index): "val" for index in val_idx})
    mapping.update({int(index): "test" for index in test_idx})
    return mapping
