"""Reusable target encoding and VOC-style evaluation for object detection.

The functions in this module are extracted from the final PASCAL VOC2012
detection experiment. They preserve the two-slot 8x8 target representation,
class-aware non-maximum suppression, and continuous VOC AP@0.50 semantics used
for model selection and final evaluation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET

import numpy as np

from . import VOC_CLASSES
from .metrics import box_iou_one_to_many

DEFAULT_GRID_SIZE = 8
DEFAULT_BOX_SLOTS = 2
DEFAULT_N_CLASSES = len(VOC_CLASSES)
DEFAULT_AP_IOU_THRESHOLD = 0.50
DEFAULT_NMS_IOU_THRESHOLD = 0.50
DEFAULT_AP_SCORE_FLOOR = 0.001
DEFAULT_MAX_DETECTIONS = 100

CLASS_TO_INDEX = {name: index for index, name in enumerate(VOC_CLASSES)}


def parse_voc_detection_record(
    image_id: str,
    voc_root: str | Path,
    split_name: str,
) -> dict[str, Any]:
    """Parse one VOC XML file into validated relative ``xywh`` boxes.

    Image dimensions from the XML annotation are checked against the JPEG before
    any target encoding is performed. Bounding boxes remain in the same relative
    ``(x, y, width, height)`` convention used by the executed experiment.
    """
    from PIL import Image

    root_dir = Path(voc_root)
    image_id = str(image_id)
    image_path = root_dir / "JPEGImages" / f"{image_id}.jpg"
    xml_path = root_dir / "Annotations" / f"{image_id}.xml"

    if not image_path.is_file():
        raise FileNotFoundError(f"Missing JPEG image: {image_path}")
    if not xml_path.is_file():
        raise FileNotFoundError(f"Missing XML annotation: {xml_path}")

    annotation = ET.parse(xml_path).getroot()
    size_node = annotation.find("size")
    if size_node is None:
        raise ValueError(f"Annotation {image_id} has no <size> node.")

    width = int(size_node.findtext("width", default="0"))
    height = int(size_node.findtext("height", default="0"))
    if width <= 0 or height <= 0:
        raise ValueError(f"Annotation {image_id} has invalid size {width}x{height}.")

    with Image.open(image_path) as image:
        jpeg_width, jpeg_height = image.size
    if (jpeg_width, jpeg_height) != (width, height):
        raise ValueError(
            f"Image/XML size mismatch for {image_id}: "
            f"JPEG={jpeg_width}x{jpeg_height}, XML={width}x{height}."
        )

    objects: list[dict[str, Any]] = []
    for object_node in annotation.findall("object"):
        label = object_node.findtext("name", default="").strip()
        if label not in CLASS_TO_INDEX:
            raise ValueError(f"Unknown VOC label {label!r} in {image_id}.")

        box_node = object_node.find("bndbox")
        if box_node is None:
            raise ValueError(f"Object in {image_id} has no bounding box.")

        xmin = float(box_node.findtext("xmin", default="nan"))
        ymin = float(box_node.findtext("ymin", default="nan"))
        xmax = float(box_node.findtext("xmax", default="nan"))
        ymax = float(box_node.findtext("ymax", default="nan"))
        raw_box = np.asarray([xmin, ymin, xmax, ymax], dtype=np.float64)

        if not np.isfinite(raw_box).all():
            raise ValueError(f"Non-finite bounding box in {image_id}: {raw_box}")
        if not (0.0 <= xmin < xmax <= width and 0.0 <= ymin < ymax <= height):
            raise ValueError(
                f"Out-of-range bounding box in {image_id}: {raw_box.tolist()} "
                f"for image {width}x{height}."
            )

        relative_x = xmin / width
        relative_y = ymin / height
        relative_width = (xmax - xmin) / width
        relative_height = (ymax - ymin) / height
        centre_x = relative_x + 0.5 * relative_width
        centre_y = relative_y + 0.5 * relative_height

        relative_values = np.asarray(
            [
                relative_x,
                relative_y,
                relative_width,
                relative_height,
                centre_x,
                centre_y,
            ],
            dtype=np.float64,
        )
        if not np.isfinite(relative_values).all():
            raise ValueError(f"Invalid relative box in {image_id}.")
        if not (
            0.0 <= relative_x < 1.0
            and 0.0 <= relative_y < 1.0
            and 0.0 < relative_width <= 1.0
            and 0.0 < relative_height <= 1.0
            and 0.0 < centre_x < 1.0
            and 0.0 < centre_y < 1.0
        ):
            raise ValueError(
                f"Relative box failed validation in {image_id}: "
                f"{relative_values.tolist()}"
            )

        objects.append(
            {
                "label": label,
                "class_index": int(CLASS_TO_INDEX[label]),
                "box_xywh": np.asarray(
                    [relative_x, relative_y, relative_width, relative_height],
                    dtype=np.float32,
                ),
                "centre_xy": np.asarray([centre_x, centre_y], dtype=np.float32),
                "area": float(relative_width * relative_height),
                "difficult": bool(
                    int(object_node.findtext("difficult", default="0"))
                ),
                "truncated": bool(
                    int(object_node.findtext("truncated", default="0"))
                ),
            }
        )

    if not objects:
        raise ValueError(f"Annotation {image_id} contains no objects.")

    return {
        "image_id": image_id,
        "split": split_name,
        "image_path": str(image_path),
        "width": width,
        "height": height,
        "objects": objects,
    }


def encode_detection_target(
    record: Mapping[str, Any],
    *,
    grid_size: int = DEFAULT_GRID_SIZE,
    box_slots: int = DEFAULT_BOX_SLOTS,
    n_classes: int = DEFAULT_N_CLASSES,
) -> tuple[np.ndarray, dict[str, int]]:
    """Encode one annotation record into the experiment's grid target.

    The final target channel is a validity flag for objectness supervision. Cells
    containing only difficult objects are ignored for objectness rather than
    treated as negatives. When more easy objects land in a cell than available
    slots, larger boxes are assigned first and the remainder are audited as
    representation collisions.
    """
    if grid_size <= 0 or box_slots <= 0 or n_classes <= 0:
        raise ValueError("grid_size, box_slots, and n_classes must be positive.")

    target_depth = 6 + n_classes
    target = np.zeros(
        (grid_size, grid_size, box_slots, target_depth),
        dtype=np.float32,
    )
    target[..., -1] = 1.0

    objects_by_cell: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    difficult_cells: set[tuple[int, int]] = set()
    n_easy = 0
    n_difficult = 0

    for obj in record["objects"]:
        centre_x, centre_y = (float(value) for value in obj["centre_xy"])
        grid_x = min(grid_size - 1, int(centre_x * grid_size))
        grid_y = min(grid_size - 1, int(centre_y * grid_size))
        cell_key = (grid_y, grid_x)

        if obj["difficult"]:
            difficult_cells.add(cell_key)
            n_difficult += 1
        else:
            objects_by_cell.setdefault(cell_key, []).append(obj)
            n_easy += 1

    encoded_easy = 0
    dropped_easy = 0
    max_easy_in_one_cell = 0

    for (grid_y, grid_x), cell_objects in objects_by_cell.items():
        cell_objects = sorted(
            cell_objects,
            key=lambda obj: (-float(obj["area"]), int(obj["class_index"])),
        )
        max_easy_in_one_cell = max(max_easy_in_one_cell, len(cell_objects))
        assigned_objects = cell_objects[:box_slots]
        dropped_easy += max(0, len(cell_objects) - box_slots)

        for slot_index, obj in enumerate(assigned_objects):
            class_index = int(obj["class_index"])
            if not 0 <= class_index < n_classes:
                raise ValueError(f"class_index {class_index} is outside 0..{n_classes - 1}.")

            centre_x, centre_y = (float(value) for value in obj["centre_xy"])
            _, _, relative_width, relative_height = (
                float(value) for value in obj["box_xywh"]
            )

            target[grid_y, grid_x, slot_index, 0] = 1.0
            target[grid_y, grid_x, slot_index, 1] = centre_x * grid_size - grid_x
            target[grid_y, grid_x, slot_index, 2] = centre_y * grid_size - grid_y
            target[grid_y, grid_x, slot_index, 3] = np.sqrt(relative_width)
            target[grid_y, grid_x, slot_index, 4] = np.sqrt(relative_height)
            target[grid_y, grid_x, slot_index, 5 + class_index] = 1.0
            encoded_easy += 1

        if (grid_y, grid_x) in difficult_cells:
            for slot_index in range(len(assigned_objects), box_slots):
                target[grid_y, grid_x, slot_index, -1] = 0.0

    for grid_y, grid_x in difficult_cells - set(objects_by_cell):
        target[grid_y, grid_x, :, -1] = 0.0

    positive_mask = target[..., 0] == 1.0
    if positive_mask.any():
        if not np.all(target[..., -1][positive_mask] == 1.0):
            raise RuntimeError("Positive slots must have valid objectness supervision.")
        if not np.allclose(target[..., 5 : 5 + n_classes][positive_mask].sum(axis=-1), 1.0):
            raise RuntimeError("Positive slots must contain exactly one class target.")
        if not np.all(target[..., 1:3][positive_mask] >= 0.0):
            raise RuntimeError("Grid offsets must be non-negative.")
        if not np.all(target[..., 1:3][positive_mask] < 1.0):
            raise RuntimeError("Grid offsets must be strictly below one.")
        if not np.all(target[..., 3:5][positive_mask] > 0.0):
            raise RuntimeError("Encoded widths/heights must be positive.")
        if not np.all(target[..., 3:5][positive_mask] <= 1.0):
            raise RuntimeError("Encoded widths/heights must not exceed one.")

    diagnostics = {
        "n_easy_boxes": n_easy,
        "n_difficult_boxes": n_difficult,
        "n_encoded_easy_boxes": encoded_easy,
        "n_dropped_easy_boxes": dropped_easy,
        "n_ignored_objectness_slots": int(np.sum(target[..., -1] == 0.0)),
        "max_easy_boxes_in_one_cell": max_easy_in_one_cell,
    }
    if encoded_easy + dropped_easy != n_easy:
        raise RuntimeError("Encoded and dropped easy boxes do not match the input count.")
    return target, diagnostics


def build_detection_targets(
    image_ids: Sequence[str],
    records_by_id: Mapping[str, Mapping[str, Any]],
    *,
    grid_size: int = DEFAULT_GRID_SIZE,
    box_slots: int = DEFAULT_BOX_SLOTS,
    n_classes: int = DEFAULT_N_CLASSES,
) -> tuple[np.ndarray, list[dict[str, int]]]:
    """Encode a sequence of detection records in image-ID order."""
    targets: list[np.ndarray] = []
    diagnostics: list[dict[str, int]] = []
    for image_id in image_ids:
        target, audit = encode_detection_target(
            records_by_id[str(image_id)],
            grid_size=grid_size,
            box_slots=box_slots,
            n_classes=n_classes,
        )
        targets.append(target)
        diagnostics.append(audit)

    if not targets:
        shape = (0, grid_size, grid_size, box_slots, 6 + n_classes)
        return np.empty(shape, dtype=np.float32), diagnostics
    return np.stack(targets).astype(np.float32), diagnostics


def sigmoid_np(values: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid used by the detector decoder."""
    values = np.asarray(values, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


def softmax_np(values: np.ndarray) -> np.ndarray:
    """Stable softmax over the final axis."""
    values = np.asarray(values, dtype=np.float32)
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=-1, keepdims=True)


def decode_and_nms(
    raw_predictions: np.ndarray,
    *,
    grid_size: int = DEFAULT_GRID_SIZE,
    box_slots: int = DEFAULT_BOX_SLOTS,
    n_classes: int = DEFAULT_N_CLASSES,
    nms_iou_threshold: float = DEFAULT_NMS_IOU_THRESHOLD,
    score_floor: float = DEFAULT_AP_SCORE_FLOOR,
    max_detections: int = DEFAULT_MAX_DETECTIONS,
) -> list[dict[str, np.ndarray]]:
    """Decode raw detector logits and apply class-aware TensorFlow NMS.

    TensorFlow is imported lazily so NumPy-only utilities and unit tests remain
    lightweight. The box transform and score calculation match the executed
    experiment: ``score = sigmoid(objectness) * softmax(class)``.
    """
    import tensorflow as tf

    raw_predictions = np.asarray(raw_predictions, dtype=np.float32)
    prediction_depth = 5 + n_classes
    expected_shape = (grid_size, grid_size, box_slots, prediction_depth)
    if raw_predictions.ndim != 5 or raw_predictions.shape[1:] != expected_shape:
        raise ValueError(f"Unexpected raw detection shape: {raw_predictions.shape}")

    n_images = raw_predictions.shape[0]
    object_probability = sigmoid_np(raw_predictions[..., 0])
    xy_offset = sigmoid_np(raw_predictions[..., 1:3])
    relative_wh = np.square(sigmoid_np(raw_predictions[..., 3:5]))
    class_probability = softmax_np(raw_predictions[..., 5 : 5 + n_classes])

    grid_y, grid_x = np.meshgrid(
        np.arange(grid_size, dtype=np.float32),
        np.arange(grid_size, dtype=np.float32),
        indexing="ij",
    )
    grid_x = grid_x[None, :, :, None]
    grid_y = grid_y[None, :, :, None]

    centre_x = (grid_x + xy_offset[..., 0]) / grid_size
    centre_y = (grid_y + xy_offset[..., 1]) / grid_size
    width = relative_wh[..., 0]
    height = relative_wh[..., 1]

    x_min = np.clip(centre_x - 0.5 * width, 0.0, 1.0)
    y_min = np.clip(centre_y - 0.5 * height, 0.0, 1.0)
    x_max = np.clip(centre_x + 0.5 * width, 0.0, 1.0)
    y_max = np.clip(centre_y + 0.5 * height, 0.0, 1.0)

    boxes_yxyx = np.stack([y_min, x_min, y_max, x_max], axis=-1)
    boxes_yxyx = boxes_yxyx.reshape(n_images, -1, 1, 4)
    scores = (
        object_probability[..., None] * class_probability
    ).reshape(n_images, -1, n_classes)

    nms = tf.image.combined_non_max_suppression(
        boxes=tf.convert_to_tensor(boxes_yxyx, dtype=tf.float32),
        scores=tf.convert_to_tensor(scores, dtype=tf.float32),
        max_output_size_per_class=50,
        max_total_size=max_detections,
        iou_threshold=nms_iou_threshold,
        score_threshold=score_floor,
        clip_boxes=True,
    )

    nms_boxes = nms.nmsed_boxes.numpy()
    nms_scores = nms.nmsed_scores.numpy()
    nms_classes = nms.nmsed_classes.numpy().astype(np.int32)
    valid_detections = nms.valid_detections.numpy().astype(np.int32)

    detections: list[dict[str, np.ndarray]] = []
    for image_index in range(n_images):
        n_valid = int(valid_detections[image_index])
        boxes = nms_boxes[image_index, :n_valid]
        boxes_xyxy = boxes[:, [1, 0, 3, 2]]
        detections.append(
            {
                "boxes": boxes_xyxy.astype(np.float32),
                "scores": nms_scores[image_index, :n_valid].astype(np.float32),
                "classes": nms_classes[image_index, :n_valid],
            }
        )
    return detections


def relative_xywh_to_xyxy(box_xywh: Sequence[float]) -> np.ndarray:
    """Convert relative ``(x, y, width, height)`` to ``(xmin, ymin, xmax, ymax)``."""
    x, y, width, height = (float(value) for value in box_xywh)
    return np.asarray([x, y, x + width, y + height], dtype=np.float32)


def prepare_voc_ground_truth(
    image_ids: Sequence[str],
    records_by_id: Mapping[str, Mapping[str, Any]],
    *,
    n_classes: int = DEFAULT_N_CLASSES,
) -> tuple[dict[int, dict[str, dict[str, np.ndarray]]], np.ndarray, np.ndarray]:
    """Build per-class VOC ground truth and counts for AP evaluation."""
    ground_truth: dict[int, dict[str, dict[str, np.ndarray]]] = {
        class_index: {} for class_index in range(n_classes)
    }
    easy_positive_count = np.zeros(n_classes, dtype=np.int64)
    difficult_count = np.zeros(n_classes, dtype=np.int64)

    for raw_image_id in image_ids:
        image_id = str(raw_image_id)
        per_class_easy = {index: [] for index in range(n_classes)}
        per_class_difficult = {index: [] for index in range(n_classes)}

        for obj in records_by_id[image_id]["objects"]:
            class_index = int(obj["class_index"])
            box = relative_xywh_to_xyxy(obj["box_xywh"])
            if obj["difficult"]:
                per_class_difficult[class_index].append(box)
                difficult_count[class_index] += 1
            else:
                per_class_easy[class_index].append(box)
                easy_positive_count[class_index] += 1

        for class_index in range(n_classes):
            ground_truth[class_index][image_id] = {
                "easy": np.asarray(
                    per_class_easy[class_index], dtype=np.float32
                ).reshape(-1, 4),
                "difficult": np.asarray(
                    per_class_difficult[class_index], dtype=np.float32
                ).reshape(-1, 4),
            }

    return ground_truth, easy_positive_count, difficult_count


def all_points_interpolated_ap(
    recall: Sequence[float],
    precision: Sequence[float],
) -> float:
    """Compute continuous all-points interpolated average precision."""
    recall_array = np.asarray(recall, dtype=np.float64)
    precision_array = np.asarray(precision, dtype=np.float64)
    if recall_array.shape != precision_array.shape:
        raise ValueError("recall and precision must have the same shape.")

    augmented_recall = np.concatenate(([0.0], recall_array, [1.0]))
    augmented_precision = np.concatenate(([0.0], precision_array, [0.0]))

    for index in range(len(augmented_precision) - 2, -1, -1):
        augmented_precision[index] = max(
            augmented_precision[index], augmented_precision[index + 1]
        )

    change_indices = np.where(
        augmented_recall[1:] != augmented_recall[:-1]
    )[0]
    return float(
        np.sum(
            (augmented_recall[change_indices + 1] - augmented_recall[change_indices])
            * augmented_precision[change_indices + 1]
        )
    )


def voc_map50(
    image_ids: Sequence[str],
    detections: Sequence[Mapping[str, np.ndarray]],
    ground_truth: Mapping[int, Mapping[str, Mapping[str, np.ndarray]]],
    positive_count: Sequence[int],
    *,
    n_classes: int = DEFAULT_N_CLASSES,
    iou_threshold: float = DEFAULT_AP_IOU_THRESHOLD,
) -> dict[str, np.ndarray | float]:
    """Compute continuous VOC AP at a fixed IoU threshold.

    Difficult ground-truth matches are ignored rather than counted as false
    positives. Duplicate predictions of the same easy object are false positives,
    matching the evaluation logic used in the final notebook.
    """
    if len(detections) != len(image_ids):
        raise ValueError("Detection count does not match image IDs.")

    positive_count = np.asarray(positive_count, dtype=np.int64)
    if positive_count.shape != (n_classes,):
        raise ValueError(f"positive_count must have shape ({n_classes},).")

    ap_per_class = np.zeros(n_classes, dtype=np.float32)
    precision_per_class = np.zeros(n_classes, dtype=np.float32)
    recall_per_class = np.zeros(n_classes, dtype=np.float32)

    for class_index in range(n_classes):
        predictions: list[tuple[float, str, np.ndarray]] = []
        for raw_image_id, image_detections in zip(image_ids, detections):
            image_id = str(raw_image_id)
            class_mask = np.asarray(image_detections["classes"]) == class_index
            for box, score in zip(
                np.asarray(image_detections["boxes"])[class_mask],
                np.asarray(image_detections["scores"])[class_mask],
            ):
                predictions.append((float(score), image_id, np.asarray(box)))
        predictions.sort(key=lambda item: item[0], reverse=True)

        matched_easy = {
            str(image_id): np.zeros(
                len(ground_truth[class_index][str(image_id)]["easy"]),
                dtype=bool,
            )
            for image_id in image_ids
        }
        true_positives: list[float] = []
        false_positives: list[float] = []

        for _, image_id, predicted_box in predictions:
            gt_entry = ground_truth[class_index][image_id]
            easy_boxes = np.asarray(gt_entry["easy"], dtype=np.float32).reshape(-1, 4)
            difficult_boxes = np.asarray(
                gt_entry["difficult"], dtype=np.float32
            ).reshape(-1, 4)
            all_boxes = np.concatenate([easy_boxes, difficult_boxes], axis=0)
            overlaps = box_iou_one_to_many(predicted_box, all_boxes)

            if overlaps.size == 0 or float(np.max(overlaps)) < iou_threshold:
                true_positives.append(0.0)
                false_positives.append(1.0)
                continue

            best_index = int(np.argmax(overlaps))
            if best_index >= len(easy_boxes):
                continue

            if not matched_easy[image_id][best_index]:
                matched_easy[image_id][best_index] = True
                true_positives.append(1.0)
                false_positives.append(0.0)
            else:
                true_positives.append(0.0)
                false_positives.append(1.0)

        if true_positives:
            cumulative_tp = np.cumsum(true_positives)
            cumulative_fp = np.cumsum(false_positives)
            recall = cumulative_tp / max(int(positive_count[class_index]), 1)
            precision = cumulative_tp / np.maximum(
                cumulative_tp + cumulative_fp, 1e-12
            )
            ap_per_class[class_index] = all_points_interpolated_ap(recall, precision)
            precision_per_class[class_index] = precision[-1]
            recall_per_class[class_index] = recall[-1]

    return {
        "map50": float(np.mean(ap_per_class)),
        "ap_per_class": ap_per_class,
        "precision_per_class": precision_per_class,
        "recall_per_class": recall_per_class,
    }
