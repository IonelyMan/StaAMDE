import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

IMAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = IMAGE_ROOT.parents[1]
for path in [str(PROJECT_ROOT), str(IMAGE_ROOT)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from phase2.common.metrics import binary_metrics
from phase2.image.datasets import AndroidDataset
from phase2.image.models import IDDCNF
from phase2.image.reports import write_json, write_metric_artifacts, write_predictions


def build_device(gpu: str) -> torch.device:
    if gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_checkpoint(model, checkpoint_path: str, device: torch.device) -> dict:
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    return checkpoint if isinstance(checkpoint, dict) else {}


def compute_image_probabilities(model, images, prior):
    _, _, z_gc, log_det = model(images)
    num_classes = model.num_classes
    latent_dim = z_gc.shape[1]

    z_expand = z_gc.unsqueeze(1)
    mu_expand = model.mu_gc.unsqueeze(0)
    log_var_clamped = torch.clamp(model.log_var_gc, min=-10.0, max=10.0)
    var_expand = torch.exp(log_var_clamped).unsqueeze(0)
    dist_sq_mahalanobis = torch.sum(((z_expand - mu_expand) ** 2) / var_expand, dim=-1)
    log_det_var = torch.sum(log_var_clamped, dim=-1).unsqueeze(0)

    log_p_z_y = -0.5 * (dist_sq_mahalanobis + log_det_var)
    log_p_x_y = log_p_z_y + log_det.unsqueeze(1)
    if prior is not None:
        log_prior = prior
    else:
        log_prior = -torch.log(torch.tensor(num_classes, dtype=torch.float, device=z_gc.device))
    joint_log_likelihood = log_p_x_y + log_prior.unsqueeze(0)
    bayes_probs = F.softmax(joint_log_likelihood / latent_dim, dim=1)
    return bayes_probs[:, 1]


def infer(args):
    warnings.filterwarnings("ignore")
    device = build_device(args.gpu)
    torch.backends.cudnn.benchmark = torch.cuda.is_available()

    dataset = AndroidDataset(args.data_path, image_size=args.image_size, belong=args.belong)
    if len(dataset) == 0:
        raise SystemExit(f"没有在 {Path(args.data_path) / args.belong} 下找到图像样本")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size * 4,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )

    prior = None
    if args.use_prior:
        prior = torch.tensor([-0.3945, -1.1210], device=device)

    model = IDDCNF(
        num_classes=args.num_classes,
        num_masks=args.num_masks,
        num_attn_layers=args.num_attn_layers,
        latent_input_dim=args.latent_input_dim,
    ).to(device)
    checkpoint = load_checkpoint(model, args.eval_ckpt, device)
    print(f"Network loaded from {args.eval_ckpt}")

    all_labels = []
    all_probs = []
    sample_infos = []
    offset = 0
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"IDDCNF inference[{args.belong}]"):
            images = images.to(device)
            probs = compute_image_probabilities(model, images, prior)
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.detach().cpu().numpy())
            batch_size = labels.shape[0]
            sample_infos.extend(dataset.get_sample_info(index) for index in range(offset, offset + batch_size))
            offset += batch_size

    y_true = np.concatenate(all_labels).astype(np.int64)
    probs = np.concatenate(all_probs).astype(np.float64)

    output_dir = Path(args.output or args.save_dir)
    prediction_dir = output_dir / "predictions"
    metrics_dir = output_dir / "metrics"
    reports_dir = output_dir / "reports"
    for directory in [prediction_dir, metrics_dir, reports_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    suffix = args.belong
    write_predictions(
        prediction_dir / f"{suffix}_predictions.csv",
        sample_infos,
        probs,
        args.threshold,
    )
    metrics = binary_metrics(y_true, probs, args.threshold)
    metrics.update(
        {
            "split": args.belong,
            "n_samples": int(len(y_true)),
            "checkpoint": str(args.eval_ckpt),
            "model_logs": checkpoint.get("logs", {}),
        }
    )
    write_json(metrics_dir / f"{suffix}_metrics.json", metrics)
    write_metric_artifacts(reports_dir, suffix, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


def main(args):
    return infer(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IDDCNF 图像模态推理")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=400, help="兼容旧脚本，推理阶段不使用")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="兼容旧脚本，推理阶段不使用")
    parser.add_argument("--num_attn_layers", type=int, default=2)
    parser.add_argument("--drop_rate", type=float, default=0.1, help="兼容旧脚本，推理阶段不使用")
    parser.add_argument("--n_cluster", type=int, default=10, help="兼容旧脚本，推理阶段不使用")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--use_prior", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--eval_ckpt", type=str, required=True)
    parser.add_argument("--patience", type=int, default=10, help="兼容旧脚本，推理阶段不使用")
    parser.add_argument("--alpha", type=float, default=1.0, help="保留旧参数，当前推理使用生成式概率")
    parser.add_argument("--image_size", type=int, nargs=2, default=(448, 448))
    parser.add_argument("--num_masks", type=int, default=32)
    parser.add_argument("--latent_input_dim", type=int, default=32)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="runs/image_model")
    parser.add_argument("--output", type=str, default=None, help="推理输出目录；默认使用 --save_dir")
    parser.add_argument("--model_name", type=str, default="best_final_model.pth")
    parser.add_argument("--log_name", type=str, default="train.log")
    parser.add_argument("--belong", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--info", type=str, default="")
    args = parser.parse_args()
    main(args)
