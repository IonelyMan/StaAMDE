from typing import Dict, List, Tuple

import numpy as np

from phase2.common.metrics import binary_metrics


def train_one_epoch(model, loader, optimizer, device) -> float:
    import torch.nn.functional as F

    model.train()
    total_loss = 0.0
    total_graphs = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch)
        loss = F.cross_entropy(logits, batch.y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu()) * batch.y.numel()
        total_graphs += int(batch.y.numel())
    return total_loss / max(total_graphs, 1)


def predict(model, loader, device) -> Tuple[List[int], np.ndarray, np.ndarray]:
    import torch

    model.eval()
    probs = []
    labels = []
    sample_indices = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            prob = torch.softmax(logits, dim=-1)[:, 1]
            probs.extend(prob.detach().cpu().numpy().tolist())
            labels.extend(batch.y.view(-1).detach().cpu().numpy().tolist())
            if hasattr(batch, "sample_idx"):
                sample_ids = batch.sample_idx
            elif hasattr(batch, "sample_index"):
                sample_ids = batch.sample_index
            else:
                raise AttributeError("batch 缺少 sample_idx，无法把预测结果映射回原样本")
            sample_indices.extend(sample_ids.view(-1).detach().cpu().numpy().tolist())
    return sample_indices, np.asarray(labels, dtype=np.int64), np.asarray(probs, dtype=np.float64)


def evaluate(model, loader, device, threshold: float) -> Dict[str, object]:
    _, y, prob = predict(model, loader, device)
    return binary_metrics(y, prob, threshold)
