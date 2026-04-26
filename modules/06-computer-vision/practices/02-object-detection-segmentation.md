# Practice 02 — Object Detection and Segmentation

## 🎯 Objectives

- Implement object detection with YOLO
- Apply instance segmentation
- Evaluate yourself with mAP and IoU
- Deploy in real time

______________________________________________________________________

## 📚 Parte 1: Exercises Guided

### Exercise 1.1: YOLOv5 Pre-trained

```python
import torch

# Load YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Inferencia
img = 'path/to/image.jpg'
results = model(img)

# Mostrar results
results.show()
results.print()

# Extraer bounding boxes
detections = results.xyxy[0]  # x1, y1, x2, y2, confidence, class
print(detections)
```

______________________________________________________________________

## 🚀 Parte 2: Exercises Proposed

### Exercise 2.1: Custom Object Detector

**Statement:**
Fine-tune YOLO on custom dataset:

- Annotate Images with bounding boxes
- Train YOLOv5
- Evaluates mAP@0.5 and mAP@0.5:0.95

### Exercise 2.2: Instance Segmentation with Mask R-CNN

**Statement:**
Usa Mask R-CNN pre-trained:

- Detect and segment instances
- Draw masks per object
- Calculate IoU

### Exercise 2.3: Real-Time Detection

**Statement:**
Procesa video frame-by-frame:

- Detect objects in each frame
- Dibuja bounding boxes
- Calculate FPS (frames per second)

### Exercise 2.4: Non-Maximum Suppression

**Statement:**
Implement NMS from cero:

- Filtra bounding boxes overlapping
- Usa IoU threshold (0.5)
- Maintains only detections with greater confidence

### Exercise 2.5: Object Tracking

**Statement:**
Implement tracker simple:

- Asigna IDs a objetos detected
- Trackea movimiento entre frames
- Visualize trayectorias

______________________________________________________________________

## ✅ Checklist

- [ ] Wear Models of object detection (YOLO, Faster R-CNN)
- [ ] Implement instance segmentation
- [ ] Calculate mAP e IoU
- [ ] Process video in real time
- [ ] Implement NMS and tracking

______________________________________________________________________

## 📚 Resources

- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)
- [Detectron2](https://github.com/facebookresearch/detectron2)
