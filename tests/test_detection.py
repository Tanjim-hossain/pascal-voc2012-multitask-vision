import numpy as np

from voc_multitask.detection import (
    all_points_interpolated_ap,
    encode_detection_target,
    prepare_voc_ground_truth,
    voc_map50,
)


def make_object(
    class_index=0,
    centre=(0.25, 0.25),
    size=(0.20, 0.10),
    difficult=False,
):
    centre_x, centre_y = centre
    width, height = size
    return {
        "class_index": class_index,
        "box_xywh": np.array(
            [centre_x - width / 2, centre_y - height / 2, width, height],
            dtype=np.float32,
        ),
        "centre_xy": np.array([centre_x, centre_y], dtype=np.float32),
        "area": float(width * height),
        "difficult": difficult,
        "truncated": False,
    }


def test_encode_detection_target_places_easy_object_in_grid_slot():
    record = {"objects": [make_object(class_index=3, centre=(0.26, 0.38))]}

    target, audit = encode_detection_target(record, n_classes=20)

    grid_x = int(0.26 * 8)
    grid_y = int(0.38 * 8)
    slot = target[grid_y, grid_x, 0]

    assert slot[0] == 1.0
    assert np.isclose(slot[1], 0.26 * 8 - grid_x)
    assert np.isclose(slot[2], 0.38 * 8 - grid_y)
    assert slot[5 + 3] == 1.0
    assert slot[-1] == 1.0
    assert audit["n_encoded_easy_boxes"] == 1
    assert audit["n_dropped_easy_boxes"] == 0


def test_difficult_only_cell_is_ignored_for_objectness():
    record = {
        "objects": [
            make_object(class_index=0, centre=(0.40, 0.40), difficult=True)
        ]
    }

    target, audit = encode_detection_target(record, n_classes=20)
    grid_x = int(0.40 * 8)
    grid_y = int(0.40 * 8)

    assert np.all(target[grid_y, grid_x, :, 0] == 0.0)
    assert np.all(target[grid_y, grid_x, :, -1] == 0.0)
    assert audit["n_difficult_boxes"] == 1
    assert audit["n_ignored_objectness_slots"] == 2


def test_two_slot_encoding_drops_smallest_collision_deterministically():
    record = {
        "objects": [
            make_object(class_index=1, centre=(0.30, 0.30), size=(0.10, 0.10)),
            make_object(class_index=2, centre=(0.30, 0.30), size=(0.30, 0.30)),
            make_object(class_index=3, centre=(0.30, 0.30), size=(0.20, 0.20)),
        ]
    }

    target, audit = encode_detection_target(record, n_classes=20)
    grid_x = int(0.30 * 8)
    grid_y = int(0.30 * 8)
    encoded_classes = np.argmax(target[grid_y, grid_x, :, 5:25], axis=-1)

    assert encoded_classes.tolist() == [2, 3]
    assert audit["n_encoded_easy_boxes"] == 2
    assert audit["n_dropped_easy_boxes"] == 1
    assert audit["max_easy_boxes_in_one_cell"] == 3


def test_all_points_interpolated_ap():
    recall = np.array([0.5, 1.0])
    precision = np.array([1.0, 0.5])
    assert np.isclose(all_points_interpolated_ap(recall, precision), 0.75)


def test_voc_map50_is_one_for_a_perfect_single_class_detection():
    record = {"objects": [make_object(class_index=0, centre=(0.5, 0.5))]}
    records = {"image-1": record}
    ground_truth, positive_count, _ = prepare_voc_ground_truth(
        ["image-1"], records, n_classes=1
    )

    box = ground_truth[0]["image-1"]["easy"][0]
    detections = [
        {
            "boxes": np.array([box], dtype=np.float32),
            "scores": np.array([0.95], dtype=np.float32),
            "classes": np.array([0], dtype=np.int32),
        }
    ]

    result = voc_map50(
        ["image-1"],
        detections,
        ground_truth,
        positive_count,
        n_classes=1,
        iou_threshold=0.50,
    )

    assert np.isclose(result["map50"], 1.0)
    assert np.isclose(result["ap_per_class"][0], 1.0)
    assert np.isclose(result["precision_per_class"][0], 1.0)
    assert np.isclose(result["recall_per_class"][0], 1.0)
