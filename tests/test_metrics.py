import numpy as np

from voc_multitask.metrics import binary_iou, box_iou_one_to_many, dice_score


def test_binary_overlap_metrics():
    y_true = np.array([[1, 1], [0, 0]])
    y_pred = np.array([[0.9, 0.8], [0.7, 0.1]])
    assert np.isclose(binary_iou(y_true, y_pred), 2 / 3)
    assert np.isclose(dice_score(y_true, y_pred), 4 / 5)


def test_box_iou():
    box = np.array([0.0, 0.0, 1.0, 1.0])
    boxes = np.array([[0.0, 0.0, 1.0, 1.0], [0.5, 0.5, 1.5, 1.5]])
    result = box_iou_one_to_many(box, boxes)
    assert np.isclose(result[0], 1.0)
    assert np.isclose(result[1], 1 / 7)
