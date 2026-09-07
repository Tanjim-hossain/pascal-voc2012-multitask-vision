from pathlib import Path

import numpy as np

from voc_multitask import VOC_CLASSES
from voc_multitask.data import (
    iterative_multilabel_train_val_split,
    labels_from_xml,
    read_voc_ids,
)


def _write_annotation(path: Path, labels: list[str]) -> None:
    objects = "".join(
        f"<object><name>{label}</name></object>" for label in labels
    )
    path.write_text(f"<annotation>{objects}</annotation>", encoding="utf-8")


def test_read_voc_ids(tmp_path):
    split = tmp_path / "train.txt"
    split.write_text("2007_000001 1\n\n2007_000002 -1\n", encoding="utf-8")
    assert read_voc_ids(split) == ["2007_000001", "2007_000002"]


def test_labels_from_xml_builds_multihot_targets_and_audit(tmp_path):
    (tmp_path / "Annotations").mkdir()
    (tmp_path / "JPEGImages").mkdir()

    _write_annotation(tmp_path / "Annotations" / "a.xml", ["cat", "person"])
    _write_annotation(
        tmp_path / "Annotations" / "b.xml",
        ["dog", "not-a-voc-class"],
    )
    (tmp_path / "JPEGImages" / "a.jpg").write_bytes(b"")
    (tmp_path / "JPEGImages" / "b.jpg").write_bytes(b"")

    labels, audit = labels_from_xml(["a", "b"], tmp_path)

    assert labels.shape == (2, len(VOC_CLASSES))
    assert labels[0, VOC_CLASSES.index("cat")] == 1.0
    assert labels[0, VOC_CLASSES.index("person")] == 1.0
    assert labels[1, VOC_CLASSES.index("dog")] == 1.0
    assert audit == {
        "missing_xml_ids": [],
        "missing_image_ids": [],
        "empty_label_ids": [],
        "unknown_labels": ["not-a-voc-class"],
    }


def test_iterative_multilabel_split_is_exact_disjoint_and_deterministic():
    labels = np.array(
        [
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 1, 1],
            [0, 0, 1],
            [1, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 1],
        ],
        dtype=np.int8,
    )

    train_a, val_a = iterative_multilabel_train_val_split(
        labels,
        val_fraction=0.20,
        seed=42,
    )
    train_b, val_b = iterative_multilabel_train_val_split(
        labels,
        val_fraction=0.20,
        seed=42,
    )

    assert len(train_a) == 8
    assert len(val_a) == 2
    assert set(train_a).isdisjoint(set(val_a))
    assert sorted(np.concatenate([train_a, val_a]).tolist()) == list(range(10))
    assert np.array_equal(train_a, train_b)
    assert np.array_equal(val_a, val_b)


def test_iterative_multilabel_split_validates_inputs():
    with np.testing.assert_raises(ValueError):
        iterative_multilabel_train_val_split(np.array([1, 0, 1]))
    with np.testing.assert_raises(ValueError):
        iterative_multilabel_train_val_split(
            np.zeros((4, 2)),
            val_fraction=0.0,
        )
