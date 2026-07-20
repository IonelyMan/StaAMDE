import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase2.common.metrics import binary_metrics
from phase2.image.dataset import DexImageDataset
from phase2.image.engine import predict
from phase2.image.model_factory import build_model
from phase2.image.reports import write_json, write_metric_artifacts, write_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ConvNeXt V2 图像模态推理")
    parser.add_argument("--input", "--data-path", "--data_path", dest="input", required=True, help="dex_images 根目录")
    parser.add_argument("--model-path", required=True, help="训练得到的 best_model.pt 或 last_model.pt")
    parser.add_argument("--output", required=True, help="推理输出目录，例如 runs/image_convnextv2")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--image-size", type=int, nargs="+", default=None, help="默认读取 checkpoint 中的 image_size")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def load_checkpoint(path: Path, device):
    import torch

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def main() -> None:
    args = parse_args()
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise SystemExit("缺少 torch，请先安装 requirements.txt 中的依赖。") from exc

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    checkpoint = load_checkpoint(Path(args.model_path), device)
    config = checkpoint.get("model_config", {})
    arch = config.get("arch") or checkpoint.get("arch") or "convnextv2_atto"
    image_size = args.image_size or config.get("image_size") or [224, 224]
    model = build_model(
        arch,
        num_classes=int(config.get("num_classes", 2)),
        drop_path_rate=float(config.get("drop_path_rate", 0.0)),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])

    dataset = DexImageDataset(Path(args.input), split=args.split, image_size=image_size, train=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device == "cuda"),
    )
    y_true, probs, sample_infos = predict(model, loader, device)

    run_dir = Path(args.output)
    prediction_dir = run_dir / "predictions"
    metrics_dir = run_dir / "metrics"
    reports_dir = run_dir / "reports"
    for directory in [prediction_dir, metrics_dir, reports_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    write_predictions(
        prediction_dir / f"{args.split}_predictions.csv",
        sample_infos,
        probs,
        args.threshold,
    )
    metrics = binary_metrics(y_true, probs, args.threshold)
    metrics.update(
        {
            "split": args.split,
            "n_samples": int(len(y_true)),
            "arch": arch,
            "checkpoint": str(args.model_path),
        }
    )
    write_json(metrics_dir / f"{args.split}_metrics.json", metrics)
    write_metric_artifacts(reports_dir, args.split, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
