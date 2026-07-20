from typing import Dict, List, Tuple

import numpy as np
import torch

from phase2.common.metrics import binary_metrics


def predict(model, loader, device) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, object]]]:
    model.eval()
    labels = []
    probs = []
    sample_infos: List[Dict[str, object]] = []
    offset = 0
    with torch.no_grad():
        for images, batch_labels in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            prob = torch.softmax(logits, dim=1)[:, 1]
            labels.extend(batch_labels.numpy().tolist())
            probs.extend(prob.detach().cpu().numpy().tolist())
            batch_size = int(batch_labels.shape[0])
            sample_infos.extend(loader.dataset.get_sample_info(index) for index in range(offset, offset + batch_size))
            offset += batch_size
    return np.asarray(labels, dtype=np.int64), np.asarray(probs, dtype=np.float64), sample_infos


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None, amp: bool = False) -> float:
    model.train()
    total_loss = 0.0
    total = 0
    use_amp = bool(amp and scaler is not None and getattr(scaler, "is_enabled", lambda: False)())
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        total_loss += float(loss.detach().cpu()) * labels.numel()
        total += int(labels.numel())
    return total_loss / max(total, 1)


def evaluate(model, loader, device, threshold: float) -> Dict[str, object]:
    y_true, probs, _ = predict(model, loader, device)
    return binary_metrics(y_true, probs, threshold)
