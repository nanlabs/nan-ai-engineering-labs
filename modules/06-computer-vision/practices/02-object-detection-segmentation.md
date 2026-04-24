# Práctica 02 — Object Detection y Segmentation

## 🎯 Objetivos

- Implementar object detection con YOLO
- Aplicar instance segmentation
- Evaluar con mAP e IoU
- Deploy en tiempo real

______________________________________________________________________

## 📚 Parte 1: Ejercicios Guiados

### Ejercicio 1.1: YOLOv5 Pre-entrenado

```python
import torch

# Cargar YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Inferencia
img = 'path/to/image.jpg'
results = model(img)

# Mostrar resultados
results.show()
results.print()

# Extraer bounding boxes
detections = results.xyxy[0]  # x1, y1, x2, y2, confidence, class
print(detections)
```

______________________________________________________________________

## 🚀 Parte 2: Ejercicios Propuestos

### Ejercicio 2.1: Custom Object Detector

**Enunciado:**
Fine-tune YOLO en dataset custom:

- Anota imágenes con bounding boxes
- Entrena YOLOv5
- Evalúa mAP@0.5 y mAP@0.5:0.95

### Ejercicio 2.2: Instance Segmentation con Mask R-CNN

**Enunciado:**
Usa Mask R-CNN pre-entrenado:

- Detecta y segmenta instancias
- Dibuja máscaras por objeto
- Calcula IoU

### Ejercicio 2.3: Real-Time Detection

**Enunciado:**
Procesa video frame-by-frame:

- Detecta objetos en cada frame
- Dibuja bounding boxes
- Calcula FPS (frames per second)

### Ejercicio 2.4: Non-Maximum Suppression

**Enunciado:**
Implementa NMS desde cero:

- Filtra bounding boxes overlapping
- Usa IoU threshold (0.5)
- Mantiene solo detecciones con mayor confidence

### Ejercicio 2.5: Object Tracking

**Enunciado:**
Implementa tracker simple:

- Asigna IDs a objetos detectados
- Trackea movimiento entre frames
- Visualiza trayectorias

______________________________________________________________________

## ✅ Checklist

- [ ] Usar modelos de object detection (YOLO, Faster R-CNN)
- [ ] Implementar instance segmentation
- [ ] Calcular mAP e IoU
- [ ] Procesar video en tiempo real
- [ ] Implementar NMS y tracking

______________________________________________________________________

## 📚 Recursos

- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)
- [Detectron2](https://github.com/facebookresearch/detectron2)
