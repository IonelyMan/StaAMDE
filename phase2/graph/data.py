import csv
import os
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from phase2.common.identity import build_sample_key
from phase2.common.splits import VALID_SPLITS, random_split_indices

GRAPH_DATA_FILENAMES = ("hetero_graph_data.pt", "hetero_graph.pt")
GRAPH_META_FILENAME = "hetero_graph_meta.json"


def load_graph(path: Path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def graph_meta_path(graph_path: Path) -> Path:
    return graph_path.with_name(GRAPH_META_FILENAME)


def load_graph_meta(path: Path) -> Dict[str, object]:
    import json

    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def to_homogeneous_graph(data):
    if hasattr(data, "edge_index") and hasattr(data, "x"):
        return data
    try:
        return data.to_homogeneous(node_attrs=["x"], add_node_type=True, add_edge_type=True)
    except TypeError:
        return data.to_homogeneous()


def resolve_workers(workers: Optional[int]) -> int:
    if workers is None:
        return 1
    if workers <= 0:
        return max(1, min(os.cpu_count() or 1, 8))
    return workers


def infer_split_from_path(path: Path, root: Path) -> Optional[str]:
    if root.name.lower() in VALID_SPLITS:
        return root.name.lower()
    try:
        parts = [part.lower() for part in path.relative_to(root).parts]
    except ValueError:
        parts = [part.lower() for part in path.parts]
    for part in parts:
        if part in VALID_SPLITS:
            return part
    return None


def infer_label_from_path(path: Path) -> int:
    for parent in path.parents:
        name = parent.name.lower()
        if name in {"mal", "malware", "1"}:
            return 1
        if name in {"benign", "leg", "normal", "0"}:
            return 0
    raise ValueError(f"无法从路径推断标签: {path}")


def normalize_index_graph_path(index_path: Path, graph_text: str) -> Path:
    graph_path = Path(graph_text)
    if graph_path.exists():
        return graph_path
    candidate = index_path.parent / graph_path
    if candidate.exists():
        return candidate
    for filename in GRAPH_DATA_FILENAMES:
        sibling = graph_path.with_name(filename)
        if sibling.exists():
            return sibling
        sibling_candidate = candidate.with_name(filename)
        if sibling_candidate.exists():
            return sibling_candidate
    return graph_path


def read_index(index_path: Path, forced_split: Optional[str] = None) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with index_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("status") not in {"ok", "skipped", ""}:
                continue
            graph_path = normalize_index_graph_path(index_path, row["graph_path"])
            if not graph_path.exists():
                continue
            split = forced_split or row.get("split") or row.get("belong") or infer_split_from_path(graph_path, index_path.parent)
            rows.append(
                {
                    "sample_id": Path(row.get("apk_name") or graph_path.parent.name).stem,
                    "apk_name": row.get("apk_name") or f"{graph_path.parent.name}.apk",
                    "graph_path": str(graph_path),
                    "meta_path": row.get("meta_path") or str(graph_meta_path(graph_path)),
                    "label": int(row["label"]),
                    "split": split.lower() if isinstance(split, str) and split.lower() in VALID_SPLITS else None,
                }
            )
    if not rows:
        raise ValueError(f"{index_path} 中没有可用图样本")
    for row in rows:
        row["sample_key"] = build_sample_key(str(row["sample_id"]), row.get("split"), int(row["label"]))
    return rows


def read_graph_root(input_dir: Path, forced_split: Optional[str] = None) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    seen_sample_dirs = set()
    graph_paths = []
    for filename in GRAPH_DATA_FILENAMES:
        for graph_path in sorted(input_dir.rglob(filename)):
            sample_dir = graph_path.parent.resolve()
            if sample_dir in seen_sample_dirs:
                continue
            seen_sample_dirs.add(sample_dir)
            graph_paths.append(graph_path)
    for graph_path in graph_paths:
        split = forced_split or infer_split_from_path(graph_path, input_dir)
        label = infer_label_from_path(graph_path)
        sample_id = graph_path.parent.name
        rows.append(
            {
                "sample_id": sample_id,
                "apk_name": f"{sample_id}.apk",
                "graph_path": str(graph_path),
                "meta_path": str(graph_meta_path(graph_path)),
                "label": label,
                "split": split,
            }
        )
    if not rows:
        raise ValueError(f"{input_dir} 下没有找到图数据文件: {', '.join(GRAPH_DATA_FILENAMES)}")
    for row in rows:
        row["sample_key"] = build_sample_key(str(row["sample_id"]), row.get("split"), int(row["label"]))
    return rows


def load_samples(
    input_dir: Optional[Path],
    index_path: Optional[Path],
    forced_split: Optional[str] = None,
) -> List[Dict[str, object]]:
    if index_path:
        return read_index(index_path, forced_split=forced_split)
    if input_dir:
        return read_graph_root(input_dir, forced_split=forced_split)
    raise ValueError("请传入 --input hetero_graphs 或 --index index.csv")


def load_train_val_samples(input_dir: Optional[Path], index_path: Optional[Path]) -> List[Dict[str, object]]:
    if index_path:
        return read_index(index_path)
    if input_dir:
        train_dir = input_dir / "train"
        val_dir = input_dir / "val"
        if train_dir.exists() and val_dir.exists():
            return read_graph_root(train_dir, forced_split="train") + read_graph_root(val_dir, forced_split="val")
        return read_graph_root(input_dir)
    raise ValueError("请传入 --input hetero_graphs 或 --index index.csv")


def split_indices_auto(
    labels: np.ndarray,
    samples: Sequence[Dict[str, object]],
    test_size: float,
    val_size: float,
    seed: int,
) -> Tuple[List[int], List[int], List[int], str]:
    splits = [str(sample.get("split")).lower() if sample.get("split") else None for sample in samples]
    split_counts = Counter(split for split in splits if split)
    if all(split_counts.get(split, 0) > 0 for split in ["train", "val", "test"]):
        train_idx = [idx for idx, split in enumerate(splits) if split == "train"]
        val_idx = [idx for idx, split in enumerate(splits) if split == "val"]
        test_idx = [idx for idx, split in enumerate(splits) if split == "test"]
        return train_idx, val_idx, test_idx, "existing"
    if split_counts.get("train", 0) > 0 and split_counts.get("val", 0) > 0:
        train_idx = [idx for idx, split in enumerate(splits) if split == "train"]
        val_idx = [idx for idx, split in enumerate(splits) if split == "val"]
        return train_idx, val_idx, [], "existing_train_val"
    train_idx, val_idx, test_idx = random_split_indices(labels, test_size, val_size, seed)
    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist(), "random"


def collect_one_metadata(task: Tuple[str, str]) -> Tuple[List[str], List[Tuple[str, str, str]]]:
    graph_path_text, meta_path_text = task
    meta = load_graph_meta(Path(meta_path_text))
    node_names = meta.get("node_names") if isinstance(meta, dict) else None
    edge_counts = meta.get("edge_counts") if isinstance(meta, dict) else None
    if isinstance(node_names, dict) and isinstance(edge_counts, dict):
        node_types = list(node_names.keys())
        edge_types = []
        for edge_text in edge_counts.keys():
            parts = str(edge_text).split("/")
            if len(parts) == 3:
                edge_types.append((parts[0], parts[1], parts[2]))
        if node_types and edge_types:
            return node_types, edge_types

    data = load_graph(Path(graph_path_text))
    return list(data.node_types), list(data.edge_types)


def collect_metadata(
    samples: Sequence[Dict[str, object]],
    workers: Optional[int] = 1,
) -> Tuple[List[str], List[Tuple[str, str, str]]]:
    node_types = set()
    edge_types = set()
    tasks = [
        (
            str(sample["graph_path"]),
            str(sample.get("meta_path") or graph_meta_path(Path(str(sample["graph_path"])))),
        )
        for sample in samples
    ]
    worker_count = resolve_workers(workers)
    if worker_count <= 1 or len(tasks) <= 1:
        metadata_rows = [collect_one_metadata(task) for task in tasks]
    else:
        chunksize = max(1, len(tasks) // (worker_count * 8))
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            metadata_rows = list(executor.map(collect_one_metadata, tasks, chunksize=chunksize))
    for sample_node_types, sample_edge_types in metadata_rows:
        node_types.update(sample_node_types)
        edge_types.update(sample_edge_types)
    return sorted(node_types), sorted(edge_types)


class GraphDataset:
    def __init__(self, samples: Sequence[Dict[str, object]], indices: Sequence[int], data_format: str = "homogeneous"):
        self.samples = list(samples)
        self.indices = list(indices)
        self.data_format = data_format

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        import torch

        sample_index = int(self.indices[item])
        sample = self.samples[sample_index]
        data = load_graph(Path(str(sample["graph_path"])))
        if self.data_format == "homogeneous":
            data = to_homogeneous_graph(data)
        data.y = torch.tensor([int(sample["label"])], dtype=torch.long)
        # Avoid names containing "index": PyG Batch treats them as node indices
        # and auto-increments them by num_nodes, which corrupts sample ids.
        data.sample_idx = torch.tensor([sample_index], dtype=torch.long)
        return data


def indices_for_split(samples: Sequence[Dict[str, object]], split: str) -> List[int]:
    if split == "all":
        return list(range(len(samples)))
    return [idx for idx, sample in enumerate(samples) if sample.get("split") == split]
