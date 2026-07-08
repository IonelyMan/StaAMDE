import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase2.graph.data import load_graph, load_graph_meta, load_samples, to_homogeneous_graph
from phase2.graph.inference import describe_graph_input, resolve_graph_inputs
from phase2.graph.model import build_model
from phase2.graph.reports import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="为 XAIDroid 风格 GATv2 图模型生成 attention 解释")
    parser.add_argument("--input", help="hetero_graphs 根目录")
    parser.add_argument("--index", help="phase1/build_heterogeneous.py 生成的 index.csv")
    parser.add_argument("--model-path", required=True, help="训练得到的 gatv2_model.pt")
    parser.add_argument("--output", required=True, help="输出目录，例如 runs/hetero_gatv2")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument(
        "--sample-id",
        action="append",
        default=[],
        help="Only explain the given sample_id. Can be passed multiple times or as a comma-separated list.",
    )
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def load_checkpoint(path: Path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def node_lookup_from_hetero(data, meta: Dict[str, object]) -> List[Dict[str, object]]:
    node_names = meta.get("node_names", {}) if isinstance(meta, dict) else {}
    lookup: List[Dict[str, object]] = []
    for node_type in data.node_types:
        num_nodes = int(data[node_type].num_nodes)
        names = node_names.get(node_type, []) if isinstance(node_names, dict) else []
        for local_index in range(num_nodes):
            if local_index < len(names):
                name = names[local_index]
            else:
                name = f"{node_type}:{local_index}"
            lookup.append(
                {
                    "node_type": node_type,
                    "local_index": local_index,
                    "node_name": name,
                }
            )
    return lookup


def attention_values(alpha) -> np.ndarray:
    values = alpha.detach().cpu().numpy()
    if values.ndim == 2:
        values = values.mean(axis=1)
    return values.reshape(-1)


def write_rows(path: Path, fieldnames: List[str], rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_requested_sample_ids(values: Sequence[str]) -> Set[str]:
    sample_ids: Set[str] = set()
    for value in values:
        for item in str(value).split(","):
            item = item.strip()
            if item:
                sample_ids.add(item)
    return sample_ids


def sample_match_keys(sample: Dict[str, object]) -> Set[str]:
    sample_id = str(sample.get("sample_id") or "")
    apk_name = str(sample.get("apk_name") or "")
    sample_key = str(sample.get("sample_key") or "")
    keys = {value for value in [sample_id, apk_name, sample_key] if value}
    if apk_name:
        keys.add(Path(apk_name).stem)
    return keys


def filter_samples_by_ids(samples: Sequence[Dict[str, object]], requested: Set[str]) -> List[Dict[str, object]]:
    if not requested:
        return list(samples)

    matched = [sample for sample in samples if sample_match_keys(sample) & requested]
    if matched:
        return matched

    available = sorted(str(sample.get("sample_id") or sample.get("apk_name") or "") for sample in samples)[:20]
    raise SystemExit(
        "No graph samples matched --sample-id. "
        f"requested={sorted(requested)}; available_examples={available}"
    )


def explain_one_sample(model, sample: Dict[str, object], device, threshold: float = 0.5):
    import torch

    hetero = load_graph(Path(str(sample["graph_path"])))
    meta = load_graph_meta(Path(str(sample.get("meta_path") or Path(str(sample["graph_path"])).with_name("hetero_graph_meta.json"))))
    lookup = node_lookup_from_hetero(hetero, meta)

    data = to_homogeneous_graph(hetero)
    data.y = torch.tensor([int(sample["label"])], dtype=torch.long)
    data = data.to(device)

    model.eval()
    with torch.no_grad():
        logits, attention = model(data, return_attention_weights=True)
        prob = torch.softmax(logits, dim=-1)[0, 1].detach().cpu().item()
        pred = int(prob >= threshold)

    edge_index, alpha = attention
    edge_index = edge_index.detach().cpu()
    edge_scores = attention_values(alpha)
    edge_rows: List[Dict[str, object]] = []
    node_scores = defaultdict(list)

    for edge_id in range(edge_index.size(1)):
        src = int(edge_index[0, edge_id])
        dst = int(edge_index[1, edge_id])
        score = float(edge_scores[edge_id]) if edge_id < len(edge_scores) else 0.0
        src_info = lookup[src] if src < len(lookup) else {"node_type": "unknown", "local_index": src, "node_name": str(src)}
        dst_info = lookup[dst] if dst < len(lookup) else {"node_type": "unknown", "local_index": dst, "node_name": str(dst)}
        node_scores[src].append(score)
        node_scores[dst].append(score)
        edge_rows.append(
            {
                "sample_id": sample["sample_id"],
                "apk_name": sample["apk_name"],
                "y_true": int(sample["label"]),
                "pred": pred,
                "prob_malware": prob,
                "source_node": src,
                "source_type": src_info["node_type"],
                "source_name": src_info["node_name"],
                "target_node": dst,
                "target_type": dst_info["node_type"],
                "target_name": dst_info["node_name"],
                "attention": score,
            }
        )

    node_rows: List[Dict[str, object]] = []
    raw_node_scores = []
    for node_id, info in enumerate(lookup):
        values = node_scores.get(node_id, [])
        score = float(np.mean(values)) if values else 0.0
        raw_node_scores.append(score)
        node_rows.append(
            {
                "sample_id": sample["sample_id"],
                "apk_name": sample["apk_name"],
                "y_true": int(sample["label"]),
                "pred": pred,
                "prob_malware": prob,
                "node": node_id,
                "node_type": info["node_type"],
                "local_index": info["local_index"],
                "node_name": info["node_name"],
                "attention": score,
            }
        )

    total = sum(raw_node_scores)
    if total > 0:
        for row in node_rows:
            row["normalized_attention"] = float(row["attention"]) / total
    else:
        for row in node_rows:
            row["normalized_attention"] = 0.0

    return edge_rows, node_rows


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    explain_dir = output_dir / "explanations"
    edge_dir = explain_dir / "edge_attention"
    node_dir = explain_dir / "node_attention"
    for directory in [edge_dir, node_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    import torch

    checkpoint = load_checkpoint(Path(args.model_path))
    config = checkpoint["model_config"]
    metadata = checkpoint["metadata"]
    model_type = checkpoint.get("model_type") or config.get("model_type")
    if model_type != "gatv2":
        raise SystemExit("当前 explain.py 只支持新 GATv2 checkpoint；旧 HGT checkpoint 没有可直接导出的 GAT attention。")

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    model = build_model(
        metadata,
        config["hidden_channels"],
        config["layers"],
        config["heads"],
        config["dropout"],
        model_type="gatv2",
        type_embedding_dim=config.get("type_embedding_dim", 16),
    ).to(device)

    input_dir, index_path, forced_split = resolve_graph_inputs(
        Path(args.input) if args.input else None,
        Path(args.index) if args.index else None,
        args.split,
    )
    try:
        samples = load_samples(input_dir, index_path, forced_split=forced_split)
    except ValueError as exc:
        raise SystemExit(f"{exc}\n{describe_graph_input(input_dir, args.split)}") from exc
    samples = [sample for sample in samples if forced_split or sample.get("split") == args.split or args.split == "all"]
    if not samples:
        raise SystemExit(describe_graph_input(input_dir, args.split))

    requested_sample_ids = normalize_requested_sample_ids(args.sample_id)
    samples = filter_samples_by_ids(samples, requested_sample_ids)
    if len(samples) > args.max_samples and not requested_sample_ids:
        rng = random.Random(args.seed)
        samples = rng.sample(samples, args.max_samples)

    first_hetero = load_graph(Path(str(samples[0]["graph_path"])))
    first = to_homogeneous_graph(first_hetero).to(device)
    with torch.no_grad():
        model(first)
    model.load_state_dict(checkpoint["state_dict"])

    all_method_rows: List[Dict[str, object]] = []
    all_class_rows: List[Dict[str, object]] = []
    processed = 0
    for sample in samples:
        edge_rows, node_rows = explain_one_sample(model, sample, device)
        sample_id = str(sample["sample_id"])
        write_rows(
            edge_dir / f"{sample_id}.csv",
            [
                "sample_id",
                "apk_name",
                "y_true",
                "pred",
                "prob_malware",
                "source_node",
                "source_type",
                "source_name",
                "target_node",
                "target_type",
                "target_name",
                "attention",
            ],
            edge_rows,
        )
        write_rows(
            node_dir / f"{sample_id}.csv",
            [
                "sample_id",
                "apk_name",
                "y_true",
                "pred",
                "prob_malware",
                "node",
                "node_type",
                "local_index",
                "node_name",
                "attention",
                "normalized_attention",
            ],
            sorted(node_rows, key=lambda row: row["attention"], reverse=True),
        )
        all_method_rows.extend(row for row in node_rows if row["node_type"] == "method")
        all_class_rows.extend(row for row in node_rows if row["node_type"] == "class")
        processed += 1

    rank_fields = [
        "sample_id",
        "apk_name",
        "y_true",
        "pred",
        "prob_malware",
        "node_name",
        "attention",
        "normalized_attention",
    ]
    write_rows(
        explain_dir / "method_attention_top.csv",
        rank_fields,
        [
            {key: row[key] for key in rank_fields}
            for row in sorted(all_method_rows, key=lambda row: row["attention"], reverse=True)[: args.top_k]
        ],
    )
    write_rows(
        explain_dir / "class_attention_top.csv",
        rank_fields,
        [
            {key: row[key] for key in rank_fields}
            for row in sorted(all_class_rows, key=lambda row: row["attention"], reverse=True)[: args.top_k]
        ],
    )
    write_json(
        explain_dir / "explain_config.json",
        {
            "model_path": str(args.model_path),
            "split": args.split,
            "sample_ids": sorted(requested_sample_ids) if requested_sample_ids else None,
            "n_samples": processed,
            "max_samples": args.max_samples,
            "top_k": args.top_k,
        },
    )
    print(f"explanations written: {explain_dir}")


if __name__ == "__main__":
    main()

"""
--sample-id com.thecybernanny.adroapp
用法：
python -m phase2.graph.explain \
  --input /home/linux/7T/lzw/datasets/android_zoo/android_graphs \
  --model-path runs/graph/07-03_15-37/models/gatv2_model.pt \
  --output ./explain/graph \
  --split test \
  --sample-id com.thecybernanny.adroapp

它会精确生成：
runs/hetero_gatv2/explanations/node_attention/com.thecybernanny.adroapp.csv
runs/hetero_gatv2/explanations/edge_attention/com.thecybernanny.adroapp.csv

也支持多个样本：
python -m phase2.graph.explain \
  --input outputs/hetero_graphs \
  --model-path runs/hetero_gatv2/models/gatv2_model.pt \
  --output runs/hetero_gatv2 \
  --split test \
  --sample-id sample001 \
  --sample-id sample002
  
或者逗号分隔：
--sample-id sample001,sample002
"""