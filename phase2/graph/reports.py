import csv
import json
from pathlib import Path
from typing import Dict, Sequence


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_predictions(
    output_path: Path,
    samples: Sequence[Dict[str, object]],
    sample_indices,
    probs,
    threshold: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_key", "sample_id", "apk_name", "split", "y_true", "prob_malware", "pred"],
        )
        writer.writeheader()
        for sample_index, prob in zip(sample_indices, probs):
            sample_index = int(sample_index)
            if sample_index < 0 or sample_index >= len(samples):
                raise IndexError(
                    f"预测结果中的 sample_idx={sample_index} 超出 samples 范围 0..{len(samples) - 1}。"
                    "如果你看到的是 sample_index 字段，说明 PyG Batch 可能把它当作节点索引自动累加了；"
                    "请使用修复后的 GraphDataset.sample_idx 重新推理。"
                )
            sample = samples[sample_index]
            writer.writerow(
                {
                    "sample_key": sample.get("sample_key"),
                    "sample_id": sample["sample_id"],
                    "apk_name": sample["apk_name"],
                    "split": sample.get("split") or "unknown",
                    "y_true": int(sample["label"]),
                    "prob_malware": float(prob),
                    "pred": int(prob >= threshold),
                }
            )
