import argparse
import json
import os
import random
import sys
import tempfile
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


def training_class_weights(labels: np.ndarray, train_idx, mode: str):
    """Compute loss weights from training labels only; keep mean sample weight at 1."""
    train_labels = labels[np.asarray(train_idx, dtype=np.int64)]
    if not np.isin(train_labels, [0, 1]).all():
        raise ValueError("图分类只支持标签 0（良性）和 1（恶意）")
    counts = np.bincount(train_labels, minlength=2)
    if (counts == 0).any():
        raise ValueError(f"训练集必须同时包含良性和恶意样本，当前类别计数: {counts.tolist()}")
    if mode == "none":
        return counts.tolist(), None
    weights = counts.sum() / (2.0 * counts.astype(np.float64))
    if mode == "sqrt":
        weights = np.sqrt(weights)
    elif mode != "balanced":
        raise ValueError(f"未知类别权重模式: {mode}")
    weights /= float(np.dot(counts, weights) / counts.sum())
    return counts.tolist(), weights.tolist()


def save_best_checkpoint(torch, path: Path, state_dict, metadata, args) -> None:
    """Atomically replace the last validated best model."""
    payload = {
        "state_dict": state_dict,
        "metadata": metadata,
        "model_type": args.model_type,
        "model_config": {
            "hidden_channels": args.hidden_channels,
            "layers": args.layers,
            "heads": args.heads,
            "dropout": args.dropout,
            "model_type": args.model_type,
            "type_embedding_dim": args.type_embedding_dim,
            "readout": args.readout,
            "node_type_order": "global" if args.model_type == "atgatv2" else "legacy",
        },
        "train_args": vars(args),
    }
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".graph-best-", suffix=".pt",
                                         delete=False) as handle:
            temporary = Path(handle.name)
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 ATGATv2 图分类器")
    parser.add_argument("--input", help="hetero_graphs 根目录；若存在 train/val 子目录，训练只读取这两个 split")
    parser.add_argument("--index", help="phase1/build_heterogeneous.py 生成的 index.csv")
    parser.add_argument("--output", required=True, help="训练运行目录，例如 runs/hetero_gatv2")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-channels", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--model-type", default="atgatv2", choices=["atgatv2", "hgt"])
    parser.add_argument("--type-embedding-dim", type=int, default=16, help="ATGATv2 同构图中的节点类型嵌入维度，0 则退化为标准 GATv2")
    parser.add_argument("--readout", choices=["mean", "mean_max"], default="mean_max", help="图级读出；mean_max 可保留少量高响应节点信号")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--class-weight", choices=["none", "sqrt", "balanced"], default="sqrt", help="只根据训练集类别频数计算损失权重")
    parser.add_argument("--grad-clip", type=float, default=2.0, help="梯度范数裁剪阈值；0 表示禁用")
    parser.add_argument("--lr-scheduler", choices=["none", "plateau"], default="plateau", help="验证指标停滞时降低学习率")
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
    train_counts, weight_values = training_class_weights(labels, train_idx, args.class_weight)
    print(f"划分来源={split_source}，训练类别计数={train_counts}，类别权重={weight_values}")
    print("扫描图元数据")
    metadata = collect_metadata([samples[i] for i in list(train_idx) + list(val_idx)], workers=worker_count)

    device = "cuda:0" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    class_weights = torch.tensor(weight_values, dtype=torch.float32, device=device) if weight_values is not None else None

    loader_kwargs = dataloader_kwargs(worker_count, args.prefetch_factor)
    data_format = "heterogeneous" if args.model_type == "hgt" else "homogeneous"
    node_types = metadata[0] if args.model_type == "atgatv2" else None
    train_loader = DataLoader(
        GraphDataset(samples, train_idx, data_format=data_format, node_types=node_types),
        batch_size=args.batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        GraphDataset(samples, val_idx, data_format=data_format, node_types=node_types),
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
        readout=args.readout,
        node_type_order="global",
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4, min_lr=1e-5)
        if args.lr_scheduler == "plateau" else None
    )

    best_val_score = -1.0
    best_state = None
    best_epoch = None
    stale_epochs = 0
    history = []
    model_filename = "hgt_model.pt" if args.model_type == "hgt" else "gatv2_model.pt"
    checkpoint_path = model_dir / model_filename

    # 开始训练模型
    print("开始训练模型")
    try:
        for epoch in tqdm(range(1, args.epochs + 1)):
            current_lr = optimizer.param_groups[0]["lr"]
            train_loss = train_one_epoch(model, train_loader, optimizer, device, class_weights, args.grad_clip)
            val_metrics = evaluate(model, val_loader, device, args.threshold)
            val_score = metric_for_best_model(val_metrics, "pr_auc")
            if scheduler is not None:
                scheduler.step(val_score)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_pr_auc": val_metrics["pr_auc"],
                    "val_roc_auc": val_metrics["roc_auc"],
                    "val_auc": val_metrics["roc_auc"],
                    "val_f1_score": val_metrics["f1_score"],
                    "val_f1": val_metrics["f1"],
                    "lr": current_lr,
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
                candidate_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                save_best_checkpoint(torch, checkpoint_path, candidate_state, metadata, args)
                best_val_score = float(val_score)
                best_state = candidate_state
                best_epoch = epoch
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= args.patience:
                    break
    except BaseException:
        if best_epoch is not None:
            print(f"训练中断；epoch {best_epoch} 的最佳模型已保存: {checkpoint_path}", file=sys.stderr)
        else:
            print("训练中断，尚无已完成验证的最佳 epoch。", file=sys.stderr)
        raise

    if best_state is not None:
        model.load_state_dict(best_state)

    print("评估模型")
    train_metrics = evaluate(model, train_loader, device, args.threshold)
    val_metrics = evaluate(model, val_loader, device, args.threshold)
    # 沿用 gatv2_model.pt 文件名，避免重命名已有模型文件；嵌入维度为 0 时对应标准 GATv2。
    summary = {
        "train": train_metrics,
        "val": val_metrics,
        "n_samples": len(samples),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test_reserved": len(test_idx),
        "split_source": split_source,
        "train_class_counts": {"benign": train_counts[0], "malware": train_counts[1]},
        "class_weight_mode": args.class_weight,
        "class_weights": weight_values,
        "readout": args.readout if args.model_type == "atgatv2" else "app",
        "node_type_order": "global" if args.model_type == "atgatv2" else None,
        "best_epoch": best_epoch,
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
    if best_state is None:
        save_best_checkpoint(torch, checkpoint_path, model.state_dict(), metadata, args)
    print(json.dumps(summary["val"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
