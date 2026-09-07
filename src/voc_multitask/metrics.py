"""Lightweight metrics used by the project and smoke tests."""
from __future__ import annotations

import numpy as np


def binary_iou(y_true, y_pred, threshold: float = 0.5) -> float:
    """Compute foreground intersection-over-union for binary masks."""
    truth = np.asarray(y_true).astype(bool)
    pred = np.asarray(y_pred) >= threshold
    intersection = np.logical_and(truth, pred).sum()
    union = np.logical_or(truth, pred).sum()
    return float(intersection / union) if union else 1.0


def dice_score(y_true, y_pred, threshold: float = 0.5) -> float:
    """Compute binary Dice/F1 overlap."""
    truth = np.asarray(y_true).astype(bool)
    pred = np.asarray(y_pred) >= threshold
    intersection = np.logical_and(truth, pred).sum()
    denominator = truth.sum() + pred.sum()
    return float(2.0 * intersection / denominator) if denominator else 1.0


def box_iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU between one xyxy box and an array of xyxy boxes."""
    box = np.asarray(box, dtype=float)
    boxes = np.asarray(boxes, dtype=float)
    if boxes.size == 0:
        return np.empty(0, dtype=float)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    box_area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    areas = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    union = box_area + areas - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
