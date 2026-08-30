# Student 3 — Defect Detection

Training and benchmarking of the three defect detection algorithms named in Section 2.4 of the
assignment documentation, on the **aligned output of the Student 1 + Student 2 pipeline**.

| Algorithm | Documentation section | Notebook | Framework |
|---|---|---|---|
| YOLOv10 | 2.4.1 | `01_yolov10_training.ipynb` | Ultralytics |
| Faster R-CNN | 2.4.2 | `02_faster_rcnn_training.ipynb` | torchvision |
| RT-DETR | 2.4.3 | `03_rtdetr_training.ipynb` | Ultralytics |

## Why the first version detected nothing

The detectors were originally trained on `Clean_Dataset` — raw boards squashed to 512×512. At
integration time the detector is not given that image. It is given

```
raw board -> Student 1 preprocess -> Student 2 align -> detector
```

which is a colour-normalised, CLAHE-enhanced, background-cropped, deskewed, aspect-preserved
picture. Training on one distribution and testing on another is **train/serve skew**, and the
model finds nothing. Two changes fix it:

1. **The training set is now built by the same code the integration calls** —
   `image_pipeline.detection_input()` — and the ground-truth boxes are carried through the same
   rotation and homography (`rotation_matrix_bound`, `transform_boxes`).
2. **`ALIGN_TARGET` was raised from 640 to 1280.** At 640 a ~70 px defect became ~15 px, too small
   to detect reliably; at 1280 it is ~30 px.

If `ALIGN_TARGET` is ever changed again, the dataset must be rebuilt and the models retrained.
Notebook 05 asserts this on startup so the mismatch cannot go unnoticed.

## Run order

```
00_data_preparation.ipynb     -> dataset/ (aligned images + YOLO/COCO labels),
                                 also refreshes Preprocessed_Dataset & Calibrated_Dataset
01_yolov10_training.ipynb     -> results/yolov10.json
02_faster_rcnn_training.ipynb -> results/faster_rcnn.json
03_rtdetr_training.ipynb      -> results/rtdetr.json
04_model_comparison.ipynb     -> results/model_comparison.csv, results/figures/*.png
05_pipeline_inference.ipynb   -> end-to-end inspection on raw boards / video
```

Notebook 00 must run first — the three training notebooks all read `dataset/dataset_info.json`,
which fixes the class order, the resolution and the train/validation/test split. Notebooks 01–03
are independent of each other. Notebook 04 uses whichever result files exist.

## Install

```bash
pip install ultralytics torch torchvision torchmetrics pycocotools pandas matplotlib opencv-python
```

Use the `tf_gpu` environment (Python 3.10, OpenCV 4.8) as the project README instructs.

If torchvision cannot download the Faster R-CNN COCO weights (`WinError 10054` — some networks
block `download.pytorch.org`), notebook 02 retries and then prints the exact folder to drop the
file into. See the notes above that cell.

## Hardware

A CUDA GPU is required in practice at 1280 px. The notebooks detect the device and fall back to
CPU with reduced epochs, but CPU training is 20–40× slower. FPS measured on CPU must not be
compared with GPU figures from the literature.

Batch sizes are set for roughly 8 GB of VRAM (YOLOv10-n 8, RT-DETR-l 4, Faster R-CNN 2). Halve
them on an out-of-memory error; lower `IMGSZ` only as a last resort, since that reintroduces the
small-object problem.

## Data

* **Source:** `PCB_DATASET/images` (full-resolution originals, which the VOC XML annotations
  match exactly) and `PCB_DATASET/rotation` (the same boards rotated by a known angle).
* **Split:** 70/20/10, stratified by defect class, decided **per board** so a board's original and
  rotated copy never end up on opposite sides of the split.
* **Boxes:** mapped from original coordinates through the rotation matrix and Student 2's
  homography; boxes that fall outside the cropped board are dropped and counted.
* Notebook 00 verifies that every XML `<size>` matches its image and that the rotation canvas is
  reproduced exactly before any of this is trusted.

## Outputs

```
dataset/   yolo/ (images, labels, data.yaml), coco/ (instances_*.json), dataset_info.json
runs/      training runs — weights, curves, confusion matrices (per model)
results/   one JSON per model, model_comparison.csv, per_class_comparison.csv,
           comparison_summary.md, figures/*.png (300 dpi, ready for the report)
outputs/   inspection results from notebook 05
```

## Changes made outside this folder

* `image_pipeline.py` — `ALIGN_TARGET` 640 → 1280; new `align_image_with_matrix`,
  `transform_points`, `transform_boxes`, `rotation_matrix_bound`, `rotate_bound_white_bg`,
  `read_rotation_angles`, `as_homography`, `compose`, `detection_input`, `align_from_source`.
  All existing function names and signatures still work.
* `Student2-Image Alignment & Calibration/Alignment&Calibration.ipynb` — the batch step called
  `process_from_folder(cls, filename)`, which always looks inside `Preprocessed_Dataset`; with
  `USE_RAW_ROTATION = True` the class folders are named `<Class>_rotation`, so every lookup missed
  and the batch silently aligned nothing. It now calls `align_from_source(...)`. The output folder
  no longer depends on the legacy `USE_GRAYSCALE` flag.
* `Preprocessed_Dataset/` and `Calibrated_Dataset/` — regenerated. They held stale artefacts
  (grayscale, and a 640×640 square) that no longer matched the code that claims to produce them.
