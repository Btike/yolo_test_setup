# Tomato YOLO — Mini Object Detection Demo (14 fotos)

Este repo documenta un demo rápido (y real) de **detección de objetos** usando **YOLO (Ultralytics)**.

La idea no fue “state of the art”, sino demostrar que puedo moverme en el pipeline completo:
**data → labeling → export → train → eval → inference → demo**, aplicado a algo que después se puede escalar a robots/RPA.

**Dataset:** 14 fotos, 1 clase (`Tomato`).

---

## Qué se construyó (resumen humano)
- Tomé **14 fotos de un tomate** con distintos ángulos/fondos.
- Etiqueté con bounding boxes en **Label Studio**.
- Exporté en formato **YOLO with Images** (para tener `images/`, `labels/`, `classes.txt`).
- Entrené en **Google Colab (GPU T4)** usando transfer learning desde `yolov8n.pt`.
- Descargué `best.pt` y corrí inferencia local en macOS (M1).

---

## Resultados rápidos
Con un dataset pequeño, obtuve métricas buenas para demo:
- `mAP50 ≈ 0.83`
- `mAP50-95 ≈ 0.63`

> Nota: con tan pocos datos, las métricas pueden variar mucho según el split train/val. El objetivo es validar que el pipeline funciona y que el modelo detecta.

---

## 1) Etiquetado con Label Studio (Docker)
Usé Docker para evitar instalar dependencias globales (sin conda/anaconda).

```bash
docker run -it --rm \
  -p 8080:8080 \
  -v "$(pwd)/mydata:/label-studio/data" \
  -v "$(pwd)/tomato_images:/images" \
  heartexlabs/label-studio:latest
````

Abrir: `http://localhost:8080`

* Template: **Object Detection (Bounding Boxes)**
* Label: `Tomato`
* Export: **YOLO with Images**

---

## 2) Entrenamiento en Google Colab (GPU)

Entrené con Ultralytics partiendo de un modelo preentrenado (transfer learning).

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # modelo base liviano: rápido para demos
model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    patience=20
)
```

### Parámetros importantes (qué significa cada uno)

* `yolov8n.pt`: “nano”, rápido para entrenar y correr (ideal para prototipos).
* `data`: ruta al `data.yaml` (clases + rutas train/val).
* `epochs`: cantidad de pasadas por el dataset. Con pocos datos, subir épocas ayuda.
* `imgsz=640`: tamaño de entrada (tradeoff entre precisión y velocidad).
* `batch=16`: imágenes por iteración (depende de memoria).
* `patience`: early stopping si no mejora (evita entrenar de más).

Al finalizar, el mejor checkpoint queda en:
`runs/detect/train/weights/best.pt`

---

## 3) Inferencia local (macOS) sin romper el Python del sistema

En macOS aparece el error `externally-managed-environment` (PEP 668), por eso uso `venv`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install ultralytics opencv-python
```

---

## 4) Inferencia sobre imágenes (outputs con bounding boxes)

```bash
yolo predict model=best.pt source=tomato_images conf=0.05 imgsz=640
```

Los resultados (imágenes con cajas) se guardan en:
`runs/detect/predict/` (o `predict2/`, `predict3/` si lo corriste varias veces)

---

## 5) Demo con webcam

```bash
yolo predict model=best.pt source=0 conf=0.05 imgsz=640
```

### ¿Qué significa este comando?

* `model=best.pt`: pesos entrenados (checkpoint “mejor” según validación).
* `source=0`: usa la **webcam** (device 0). También puede ser un video o una carpeta.
* `conf=0.05`: umbral mínimo de confianza. Con dataset chico conviene bajarlo para subir recall.
* `imgsz=640`: tamaño de entrada para inferencia.

> Nota (macOS): a veces la ventana “en vivo” no se muestra por temas de GUI/OpenCV.
> El output guardado (video/imagenes en `runs/…`) es el camino más confiable para demo.

---

## Próximos pasos (si esto fuera a producción)

* Subir dataset a 100–300 imágenes y aumentar variación (iluminación, fondos, oclusiones).
* Mejorar recall/robustez con más data y hard cases.
* Exportar a ONNX/TensorRT según target (pipeline robótica).

---

## Disclaimer

Este demo usa solo **14 fotos y 1 clase**, intencionalmente, para mostrar ejecución rápida del pipeline completo.


