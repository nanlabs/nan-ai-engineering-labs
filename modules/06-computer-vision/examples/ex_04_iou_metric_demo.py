"""Intersection over Union (IoU) metric demo.

Run:
    python modules/06-computer-vision/examples/ex_04_iou_metric_demo.py
"""

from __future__ import annotations


Box = tuple[float, float, float, float]


def area(box: Box) -> float:
    """Compute area for (x1, y1, x2, y2)."""
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def iou(box_a: Box, box_b: Box) -> float:
    """Compute IoU between two boxes."""
    inter_x1 = max(box_a[0], box_b[0])
    inter_y1 = max(box_a[1], box_b[1])
    inter_x2 = min(box_a[2], box_b[2])
    inter_y2 = min(box_a[3], box_b[3])

    inter_box = (inter_x1, inter_y1, inter_x2, inter_y2)
    inter_area = area(inter_box)
    union = area(box_a) + area(box_b) - inter_area
    return inter_area / union if union > 0 else 0.0


def main() -> None:
    """Compare overlap quality for two detections."""
    gt_box = (10, 10, 40, 40)
    pred_good = (12, 12, 38, 38)
    pred_bad = (25, 25, 50, 50)

    print(f"iou_good={iou(gt_box, pred_good):.4f}")
    print(f"iou_bad={iou(gt_box, pred_bad):.4f}")


if __name__ == "__main__":
    main()
