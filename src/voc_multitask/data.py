"""PASCAL VOC split and multi-label annotation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
import xml.etree.ElementTree as ET

import numpy as np

from . import VOC_CLASSES

CLASS_TO_INDEX = {name: index for index, name in enumerate(VOC_CLASSES)}


def read_voc_ids(file_path: str | Path) -> list[str]:
    """Read non-empty VOC image identifiers from a split file."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Required split file is missing: {path}")
    return [
        line.split()[0]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def labels_from_xml(
    image_ids: Iterable[str],
    voc_root: str | Path,
) -> tuple[np.ndarray, dict[str, list[str]]]:
    """Build a multi-hot classification target matrix from VOC XML annotations.

    The returned audit reports missing annotation/image files, images without any
    recognised VOC class, and labels outside the canonical 20-class vocabulary.
    """
    ids = [str(image_id) for image_id in image_ids]
    root_dir = Path(voc_root)

    label_matrix = np.zeros((len(ids), len(VOC_CLASSES)), dtype=np.float32)
    missing_xml_ids: list[str] = []
    missing_image_ids: list[str] = []
    empty_label_ids: list[str] = []
    unknown_labels: set[str] = set()

    for row_index, image_id in enumerate(ids):
        xml_path = root_dir / "Annotations" / f"{image_id}.xml"
        image_path = root_dir / "JPEGImages" / f"{image_id}.jpg"

        if not xml_path.is_file():
            missing_xml_ids.append(image_id)
            continue
        if not image_path.is_file():
            missing_image_ids.append(image_id)

        annotation = ET.parse(xml_path).getroot()
        image_classes = {
            obj.findtext("name", default="").strip()
            for obj in annotation.findall("object")
        }

        for class_name in image_classes:
            if class_name in CLASS_TO_INDEX:
                label_matrix[row_index, CLASS_TO_INDEX[class_name]] = 1.0
            elif class_name:
                unknown_labels.add(class_name)

        if label_matrix[row_index].sum() == 0:
            empty_label_ids.append(image_id)

    audit = {
        "missing_xml_ids": missing_xml_ids,
        "missing_image_ids": missing_image_ids,
        "empty_label_ids": empty_label_ids,
        "unknown_labels": sorted(unknown_labels),
    }
    return label_matrix, audit


def iterative_multilabel_train_val_split(
    labels: np.ndarray,
    val_fraction: float = 0.20,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a deterministic two-way iterative multi-label split.

    Rare labels are allocated first while exact train/validation capacities are
    enforced. The implementation mirrors the split logic used in the original
    experiment and does not require an external iterative-stratification package.
    """
    labels = np.asarray(labels, dtype=np.int8)
    if labels.ndim != 2:
        raise ValueError("labels must be a two-dimensional multi-hot matrix.")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be strictly between 0 and 1.")

    n_samples = labels.shape[0]
    target_val_size = int(round(n_samples * val_fraction))
    target_sizes = np.array(
        [n_samples - target_val_size, target_val_size],
        dtype=np.int64,
    )
    remaining_capacity = target_sizes.copy()

    total_label_counts = labels.sum(axis=0).astype(np.float64)
    desired_label_counts = np.vstack(
        [
            total_label_counts * (1.0 - val_fraction),
            total_label_counts * val_fraction,
        ]
    )

    rng = np.random.default_rng(seed)
    assignment = np.full(n_samples, -1, dtype=np.int8)
    unassigned = np.ones(n_samples, dtype=bool)

    while unassigned.any():
        remaining_label_counts = labels[unassigned].sum(axis=0)
        available_labels = np.flatnonzero(remaining_label_counts > 0)
        if available_labels.size == 0:
            break

        rarest_label = available_labels[
            np.argmin(remaining_label_counts[available_labels])
        ]
        candidate_indices = np.flatnonzero(
            unassigned & (labels[:, rarest_label] == 1)
        )
        rng.shuffle(candidate_indices)

        for sample_index in candidate_indices:
            if not unassigned[sample_index]:
                continue

            available_splits = np.flatnonzero(remaining_capacity > 0)
            label_need = desired_label_counts[available_splits, rarest_label]
            best_splits = available_splits[
                np.isclose(label_need, label_need.max())
            ]

            if best_splits.size > 1:
                capacity_ratio = (
                    remaining_capacity[best_splits] / target_sizes[best_splits]
                )
                best_splits = best_splits[
                    np.isclose(capacity_ratio, capacity_ratio.max())
                ]

            chosen_split = int(rng.choice(best_splits))
            assignment[sample_index] = chosen_split
            unassigned[sample_index] = False
            remaining_capacity[chosen_split] -= 1
            desired_label_counts[chosen_split] -= labels[sample_index]

    leftover_indices = np.flatnonzero(unassigned)
    rng.shuffle(leftover_indices)
    for sample_index in leftover_indices:
        available_splits = np.flatnonzero(remaining_capacity > 0)
        capacity_ratio = (
            remaining_capacity[available_splits] / target_sizes[available_splits]
        )
        best_splits = available_splits[
            np.isclose(capacity_ratio, capacity_ratio.max())
        ]
        chosen_split = int(rng.choice(best_splits))
        assignment[sample_index] = chosen_split
        remaining_capacity[chosen_split] -= 1

    if np.any(assignment < 0) or np.any(remaining_capacity != 0):
        raise RuntimeError("Iterative stratification did not allocate every sample.")

    train_indices = np.flatnonzero(assignment == 0)
    val_indices = np.flatnonzero(assignment == 1)
    return train_indices, val_indices
