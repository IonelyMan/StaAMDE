from typing import Dict, List, Tuple

import numpy as np

from phase2.common.metrics import binary_metrics


def train_one_epoch(model, loader, optimizer, device, class_weights=None,
                    grad_clip: float = 0.0, accumulation_steps: int = 1,
                    batch_stats=None) -> float:
    import torch
    import torch.nn.functional as F

    model.train()
    total_loss = 0.0
    total_graphs = 0
    group_loss = None
    group_graphs = 0
    pending = 0
    optimizer.zero_grad(set_to_none=True)

    def finish_group():
        nonlocal total_loss, total_graphs, group_loss, group_graphs, pending
        if not pending:
            return
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(group_graphs)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        total_loss += float(group_loss)
        total_graphs += group_graphs
        group_loss = None
        group_graphs = pending = 0

    for batch in loader:
        if batch_stats is not None:
            edge_count = int(getattr(batch, "num_edges", 0))
            if edge_count > batch_stats.get("max_edges", -1):
                sample_idx = getattr(batch, "sample_idx", None)
                batch_stats.update({
                    "max_edges": edge_count,
                    "max_nodes": int(getattr(batch, "num_nodes", 0)),
                    "sample_indices": sample_idx.view(-1).tolist() if sample_idx is not None else [],
                })
        batch = batch.to(device)
        logits = model(batch)
        labels = batch.y.view(-1)
        loss_sum = F.cross_entropy(logits, labels, weight=class_weights, reduction="sum")
        loss_sum.backward()
        group_loss = loss_sum.detach() if group_loss is None else group_loss + loss_sum.detach()
        group_graphs += int(labels.numel())
        pending += 1
        del batch, logits, loss_sum, labels
        if pending == accumulation_steps:
            finish_group()
    finish_group()
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
