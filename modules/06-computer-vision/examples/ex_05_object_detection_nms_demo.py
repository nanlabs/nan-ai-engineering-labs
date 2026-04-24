"""Non-maximum suppression (NMS) demo for object detection.

Run:
    python modules/06-computer-vision/examples/ex_05_object_detection_nms_demo.py
"""

from __future__ import annotations


BoxScore = tuple[tuple[float, float, float, float], float]


def area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    inter = area((ix1, iy1, ix2, iy2))
    union = area(box_a) + area(box_b) - inter
    return inter / union if union else 0.0


def nms(boxes: list[BoxScore], threshold: float) -> list[BoxScore]:
    """Keep highest-score boxes and suppress heavy overlaps."""
    sorted_boxes = sorted(boxes, key=lambda item: item[1], reverse=True)
    kept: list[BoxScore] = []

    while sorted_boxes:
        candidate = sorted_boxes.pop(0)
        kept.append(candidate)
        sorted_boxes = [
            item for item in sorted_boxes if iou(candidate[0], item[0]) < threshold
        ]

    return kept


def main() -> None:
    """Run NMS on overlapping detections."""
    detections: list[BoxScore] = [
        ((10, 10, 40, 40), 0.95),
        ((12, 12, 39, 39), 0.90),
        ((45, 45, 70, 70), 0.80),
        ((47, 47, 69, 69), 0.75),
    ]

    kept = nms(detections, threshold=0.5)
    print(f"input_detections={len(detections)}")
    print(f"kept_after_nms={len(kept)}")
    for box, score in kept:
        print(f"score={score:.2f} box={box}")


if __name__ == "__main__":
    main()
