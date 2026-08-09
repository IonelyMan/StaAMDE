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
from phase2.image.dataset import DexImageDataset
from phase2.image.engine import evaluate, train_one_epoch
from phase2.image.model_factory import available_models, build_model
from phase2.image.reports import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 ConvNeXt V2 图像模态分类器")
    parser.add_argument("--input", "--data-path", "--data_path", dest="input", required=True, help="dex_images 根目录")
    parser.add_argument("--output", required=True, help="训练输出目录，例如 runs/image_convnextv2")
    parser.add_argument("--arch", default="convnextv2_atto", choices=available_models())
    parser.add_argument("--image-size", type=int, nargs="+", default=[224, 224])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--drop-path-rate", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--class-weight", action="store_true", help="按训练集类别频次加权 CrossEntropy")
    parser.add_argument("--info", type=str, default="")

    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def class_weights(labels, device):
    import torch

    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=2).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def main() -> None:
    args = parse_args()
    print("本次训练的信息：",args.info)
    seed_everything(args.seed)
    run_dir = Path(args.output)
    model_dir = run_dir / "models"
    metrics_dir = run_dir / "metrics"
    logs_dir = run_dir / "logs"
    for directory in [model_dir, metrics_dir, logs_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise SystemExit("缺少 torch，请先安装 requirements.txt 中的依赖。") from exc

    device = "cuda:0" if args.device == "auto" and torch.cuda.is_available() else args.device
    print("使用GPU？：",torch.cuda.is_available())
    # if device == "auto":
    #     device = "cpu"
    print("[1] 正在加载数据集")
    train_dataset = DexImageDataset(Path(args.input), split="train", image_size=args.image_size, train=True)
    val_dataset = DexImageDataset(Path(args.input), split="val", image_size=args.image_size, train=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device == "cuda"),
    )
    print("[2] 正在创建模型")
    model = build_model(args.arch, num_classes=2, drop_path_rate=args.drop_path_rate).to(device)
    train_labels = [sample.label for sample in train_dataset.samples]
    criterion_weight = class_weights(train_labels, device) if args.class_weight else None
    criterion = torch.nn.CrossEntropyLoss(weight=criterion_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device == "cuda")
    print("[3] 开始训练")
    best_score = -1.0
    stale_epochs = 0
    history = []
    best_path = model_dir / "best_model.pt"

    for epoch in tqdm(range(1, args.epochs + 1),desc="正在训练中"):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler=scaler, amp=args.amp)
        val_metrics = evaluate(model, val_loader, device, args.threshold)
        score = metric_for_best_model(val_metrics, "pr_auc")
        scheduler.step()
        record = {
            "epoch": epoch,
            "train_loss": loss,
            "lr": optimizer.param_groups[0]["lr"],
            "val": val_metrics,
        }
        history.append(record)
        print(
            f"epoch={epoch:03d} loss={loss:.4f} "
            f"val_pr_auc={val_metrics['pr_auc']} val_roc_auc={val_metrics['roc_auc']} "
            f"val_f1={val_metrics['f1_score']:.4f}"
        )
        if score > best_score:
            best_score = float(score)
            stale_epochs = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "arch": args.arch,
                    "model_config": {
                        "arch": args.arch,
                        "num_classes": 2,
                        "drop_path_rate": args.drop_path_rate,
                        "image_size": list(args.image_size),
                    },
                    "train_args": vars(args),
                    "best_metric": "pr_auc",
                    "best_score": best_score,
                    "epoch": epoch,
                },
                best_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"early stop at epoch={epoch}, best_pr_auc={best_score}")
                break

    torch.save(
        {
            "state_dict": model.state_dict(),
            "arch": args.arch,
            "model_config": {
                "arch": args.arch,
                "num_classes": 2,
                "drop_path_rate": args.drop_path_rate,
                "image_size": list(args.image_size),
            },
            "train_args": vars(args),
            "epoch": history[-1]["epoch"] if history else 0,
        },
        model_dir / "last_model.pt",
    )
    final_val = evaluate(model, val_loader, device, args.threshold)
    summary = {
        "arch": args.arch,
        "n_train": len(train_dataset),
        "n_val": len(val_dataset),
        "best_metric": "pr_auc",
        "best_score": best_score,
        "val": final_val,
    }
    write_json(metrics_dir / "train_metrics.json", summary)
    write_json(logs_dir / "history.json", {"history": history})
    write_json(logs_dir / "train_config.json", vars(args))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
