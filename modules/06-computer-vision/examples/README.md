# Examples — Computer Vision

## Example 1 — Image loading and visualization

Read an image dataset and visualize sample images by class.

## Example 2 — Baseline preprocessing

Apply resize, normalization, and a train/validation/test split.

## Example 3 — Minimal CNN

Train a small CNN and report accuracy.

## Example 4 — error analysis

Use a confusion matrix to find hard classes.

## Rules

- Explain the input, output, and success criteria.
- Keep training reproducible.

## Available examples

### Executable scripts (phase-2 pilot)

1. `ex_01_image_matrix_basics.py`

   - Basic operations on a grayscale image matrix.
   - Expected output: min/max/mean and a normalized pixel value.

1. `ex_02_convolution_baseline.py`

   - Valid 2D convolution with an edge-detection-style kernel.
   - Expected output: the convolution output matrix.

1. `ex_03_edge_detection_filter.py`

   - Sobel-like filters for X and Y gradients.
   - Expected output: differentiated gradient maps.

1. `ex_04_iou_metric_demo.py`

   - IoU calculation for detection boxes.
   - Expected output: `iou_good` higher than `iou_bad`.

1. `ex_05_object_detection_nms_demo.py`

   - Non-maximum suppression (NMS) on overlapping detections.
   - Expected output: fewer boxes after NMS.

1. `ex_06_cv_pipeline_reproducible.py`

   - Minimal reproducible pipeline with deterministic augmentation.
   - Expected output: `same_result=True` with the same seed.

## How to use these examples

```bash
python modules/06-computer-vision/examples/ex_01_image_matrix_basics.py
python modules/06-computer-vision/examples/ex_02_convolution_baseline.py
python modules/06-computer-vision/examples/ex_03_edge_detection_filter.py
python modules/06-computer-vision/examples/ex_04_iou_metric_demo.py
python modules/06-computer-vision/examples/ex_05_object_detection_nms_demo.py
python modules/06-computer-vision/examples/ex_06_cv_pipeline_reproducible.py
```

Recommendation: run `01-03` before `04-05` to reinforce the representation and filter basics.

## Next steps

1. Add a simplified mAP example for detection evaluation.
1. Integrate NMS and IoU into an end-to-end pipeline practice.
1. Record common box and filter errors in `notes/README.md`.
