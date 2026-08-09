import argparse
import json
import random
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase2.common.metrics import metric_for_best_model
from phase2.graph.data import GraphDataset, collect_metadata, load_train_val_samples, resolve_workers, split_indices_auto
from phase2.graph.engine import evaluate, train_one_epoch
from phase2.graph.model import build_model
from phase2.graph.reports import write_json


def format_metric(value) -> str:
    return "None" if value is None else f"{float(value):.4f}"


def dataloader_kwargs(worker_count: int, prefetch_factor: int) -> dict:
    if worker_count <= 0:
        return {"num_workers": 0}
    return {
        "num_workers": worker_count,
        "persistent_workers": True,
        "prefetch_factor": prefetch_factor,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 TA-SGATv2 图分类器")
    parser.add_argument("--input", help="hetero_graphs 根目录；若存在 train/val 子目录，训练只读取这两个 split")
    parser.add_argument("--index", help="phase1/build_heterogeneous.py 生成的 index.csv")
    parser.add_argument("--output", required=True, help="训练运行目录，例如 runs/hetero_gatv2")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-channels", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--model-type", default="ta-sgatv2", choices=["ta-sgatv2", "hgt"])
    parser.add_argument("--type-embedding-dim", type=int, default=16, help="TA-SGATv2 同构图中的节点类型嵌入维度，0则退化为标准GATv2")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--workers", type=int, default=4, help="图文件并行加载进程数；0 表示自动，1 表示串行")
    parser.add_argument("--prefetch-factor", type=int, default=2, help="DataLoader 每个 worker 预取 batch 数")
    parser.add_argument("--info", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.output)
    print("本次训练信息：",args.info)
    model_dir = run_dir / "models"
    metrics_dir = run_dir / "metrics"
    logs_dir = run_dir / "logs"
    for directory in [model_dir, metrics_dir, logs_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        from torch_geometric.loader import DataLoader
    except ImportError as exc:
        raise SystemExit("缺少 torch/torch-geometric，请先安装图模型依赖。") from exc

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    worker_count = resolve_workers(args.workers)
    print(f"加载数据集，workers={worker_count}")
    samples = load_train_val_samples(Path(args.input) if args.input else None, Path(args.index) if args.index else None)
    labels = np.asarray([int(sample["label"]) for sample in samples], dtype=np.int64)
    train_idx, val_idx, test_idx, split_source = split_indices_auto(
        labels,
        samples,
        args.test_size,
        args.val_size,
        args.seed,
    )
    print("扫描图元数据")
    metadata = collect_metadata(samples, workers=worker_count)

    device = "cuda:0" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    loader_kwargs = dataloader_kwargs(worker_count, args.prefetch_factor)
    data_format = "heterogeneous" if args.model_type == "hgt" else "homogeneous"
    train_loader = DataLoader(
        GraphDataset(samples, train_idx, data_format=data_format),
        batch_size=args.batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        GraphDataset(samples, val_idx, data_format=data_format),
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    print("创建模型")
    model = build_model(
        metadata,
        args.hidden_channels,
        args.layers,
        args.heads,
        args.dropout,
        model_type=args.model_type,
        type_embedding_dim=args.type_embedding_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_score = -1.0
    best_state = None
    stale_epochs = 0
    history = []

    # 开始训练模型
    print("开始训练模型")
    for epoch in tqdm(range(1, args.epochs + 1)):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device, args.threshold)
        val_score = metric_for_best_model(val_metrics, "pr_auc")
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_pr_auc": val_metrics["pr_auc"],
                "val_roc_auc": val_metrics["roc_auc"],
                "val_auc": val_metrics["roc_auc"],
                "val_f1_score": val_metrics["f1_score"],
                "val_f1": val_metrics["f1"],
            }
        )
        print(
            "epoch={epoch:03d} loss={loss:.4f} val_pr_auc={pr_auc} val_roc_auc={roc_auc} val_f1={f1}".format(
                epoch=epoch,
                loss=train_loss,
                pr_auc=format_metric(val_metrics["pr_auc"]),
                roc_auc=format_metric(val_metrics["roc_auc"]),
                f1=format_metric(val_metrics["f1_score"]),
            )
        )

        if val_score is not None and val_score > best_val_score:
            best_val_score = float(val_score)
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    print("评估模型")
    train_metrics = evaluate(model, train_loader, device, args.threshold)
    val_metrics = evaluate(model, val_loader, device, args.threshold)
    # 由于ta-sgatv2名字太长，干脆保存时就保留gatv2_model,后续按嵌入维度=0区分标准GATv2
    model_filename = "hgt_model.pt" if args.model_type == "hgt" else "gatv2_model.pt"
    summary = {
        "train": train_metrics,
        "val": val_metrics,
        "n_samples": len(samples),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test_reserved": len(test_idx),
        "split_source": split_source,
        "threshold": float(args.threshold),
        "best_model_metric": "pr_auc",
        "best_val_score": best_val_score,
        "model_type": args.model_type,
        "model_filename": model_filename,
        "workers": worker_count,
        "prefetch_factor": args.prefetch_factor,
        "metadata": {
            "node_types": metadata[0],
            "edge_types": [list(edge_type) for edge_type in metadata[1]],
        },
    }

    print("保存相关结果")
    write_json(metrics_dir / "train_metrics.json", summary)
    write_json(logs_dir / "history.json", {"history": history})
    write_json(logs_dir / "train_config.json", vars(args))
    torch.save(
        {
            "state_dict": model.state_dict(),
            "metadata": metadata,
            "model_type": args.model_type,
            "model_config": {
                "hidden_channels": args.hidden_channels,
                "layers": args.layers,
                "heads": args.heads,
                "dropout": args.dropout,
                "model_type": args.model_type,
                "type_embedding_dim": args.type_embedding_dim,
            },
            "train_args": vars(args),
        },
        model_dir / model_filename,
    )
    print(json.dumps(summary["val"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
