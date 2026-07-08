import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase2.common.metrics import binary_metrics
from phase2.graph.data import GraphDataset, indices_for_split, load_samples, resolve_workers
from phase2.graph.engine import predict
from phase2.graph.model import build_model
from phase2.graph.reports import write_json, write_predictions


def dataloader_kwargs(worker_count: int, prefetch_factor: int) -> dict:
    if worker_count <= 0:
        return {"num_workers": 0}
    return {
        "num_workers": worker_count,
        "persistent_workers": True,
        "prefetch_factor": prefetch_factor,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用已训练图模型进行推理")
    parser.add_argument("--input", help="hetero_graphs 根目录")
    parser.add_argument("--index", help="phase1/build_heterogeneous.py 生成的 index.csv")
    parser.add_argument("--model-path", required=True, help="训练得到的 gatv2_model.pt 或 hgt_model.pt")
    parser.add_argument("--output", required=True, help="推理输出目录，例如 runs/hetero_gatv2")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--workers", type=int, default=4, help="图文件并行加载进程数；0 表示自动，1 表示串行")
    parser.add_argument("--prefetch-factor", type=int, default=2, help="DataLoader 每个 worker 预取 batch 数")
    return parser.parse_args()


def resolve_graph_inputs(
    input_dir: Optional[Path],
    index_path: Optional[Path],
    split: str,
) -> Tuple[Optional[Path], Optional[Path], Optional[str]]:
    if index_path:
        return input_dir, index_path, None
    if input_dir is None:
        return None, None, None

    if split == "all":
        root_index = input_dir / "index.csv"
        return input_dir, root_index if root_index.exists() else None, None

    if input_dir.name.lower() == split:
        split_index = input_dir / "index.csv"
        return input_dir, split_index if split_index.exists() else None, split

    split_dir = input_dir / split
    if split_dir.exists():
        split_index = split_dir / "index.csv"
        return split_dir, split_index if split_index.exists() else None, split

    root_index = input_dir / "index.csv"
    if root_index.exists():
        return input_dir, root_index, None

    return input_dir, None, None


def describe_graph_input(input_dir: Optional[Path], split: str) -> str:
    if input_dir is None:
        return "未传入 --input"
    children = []
    if input_dir.exists():
        children = [path.name for path in sorted(input_dir.iterdir())[:20]]
    return (
        f"没有找到 split={split} 的图样本，实际扫描目录: {input_dir}。"
        f"请确认目录形如 input/{split}/mal/<sample>/hetero_graph_data.pt，"
        f"或存在 input/{split}/index.csv。当前目录前 20 个子项: {children}"
    )


def main() -> None:
    args = parse_args()
    run_dir = Path(args.output)
    prediction_dir = run_dir / "predictions"
    metrics_dir = run_dir / "metrics"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        from torch_geometric.loader import DataLoader
    except ImportError as exc:
        raise SystemExit("缺少 torch/torch-geometric，请先安装图模型依赖。") from exc

    input_dir, index_path, forced_split = resolve_graph_inputs(
        Path(args.input) if args.input else None,
        Path(args.index) if args.index else None,
        args.split,
    )
    try:
        samples = load_samples(
            input_dir,
            index_path,
            forced_split=forced_split,
        )
    except ValueError as exc:
        raise SystemExit(f"{exc}\n{describe_graph_input(input_dir, args.split)}") from exc
    indices = list(range(len(samples))) if forced_split else indices_for_split(samples, args.split)
    if not indices:
        raise SystemExit(describe_graph_input(input_dir, args.split))
    worker_count = resolve_workers(args.workers)

    try:
        checkpoint = torch.load(args.model_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.model_path, map_location="cpu")
    config = checkpoint["model_config"]
    metadata = checkpoint["metadata"]
    model_type = checkpoint.get("model_type") or config.get("model_type") or "hgt"
    data_format = "heterogeneous" if model_type == "hgt" else "homogeneous"

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    loader = DataLoader(
        GraphDataset(samples, indices, data_format=data_format),
        batch_size=args.batch_size,
        shuffle=False,
        **dataloader_kwargs(worker_count, args.prefetch_factor),
    )
    model = build_model(
        metadata,
        config["hidden_channels"],
        config["layers"],
        config["heads"],
        config["dropout"],
        model_type=model_type,
        type_embedding_dim=config.get("type_embedding_dim", 16),
    ).to(device)

    first_batch = next(
        iter(DataLoader(GraphDataset(samples, indices[:1], data_format=data_format), batch_size=1, shuffle=False))
    ).to(device)
    with torch.no_grad():
        model(first_batch)
    model.load_state_dict(checkpoint["state_dict"])

    sample_indices, y, probs = predict(model, loader, device)
    suffix = args.split if args.split != "all" else "all"
    write_predictions(
        prediction_dir / f"{suffix}_predictions.csv",
        samples,
        sample_indices,
        probs,
        args.threshold,
    )
    metrics = binary_metrics(y, probs, args.threshold)
    metrics.update({"split": args.split, "n_samples": int(len(y))})
    write_json(metrics_dir / f"{suffix}_metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
