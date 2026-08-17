# Student 3 — Defect Detection

Training and benchmarking of the three defect detection algorithms named in Section 2.4 of the
assignment documentation, on `Clean_Dataset` (693 PCB images, 2 953 boxes, 6 defect classes).

| Algorithm | Documentation section | Notebook | Framework |
|---|---|---|---|
| YOLOv10 | 2.4.1 | `01_yolov10_training.ipynb` | Ultralytics |
| Faster R-CNN | 2.4.2 | `02_faster_rcnn_training.ipynb` | torchvision |
| RT-DETR | 2.4.3 | `03_rtdetr_training.ipynb` | Ultralytics |

## Run order

```
00_data_preparation.ipynb   -> dataset/  (YOLO + COCO exports, fixed 70/20/10 split)
01_yolov10_training.ipynb   -> results/yolov10.json
02_faster_rcnn_training.ipynb -> results/faster_rcnn.json
03_rtdetr_training.ipynb    -> results/rtdetr.json
04_model_comparison.ipynb   -> results/model_comparison.csv, results/figures/*.png
```

Notebook 00 must run first — the three training notebooks all read `dataset/dataset_info.json`,
which fixes the class order and the train/validation/test split so the comparison is fair.
Notebooks 01–03 are independent of each other and can be run in any order (or skipped); notebook 04
uses whichever result files exist.

## Install

```bash
pip install ultralytics torch torchvision torchmetrics pycocotools pandas matplotlib
```

Each training notebook has the matching `%pip install` line commented out in its first code cell.

## Hardware

A CUDA GPU is strongly recommended. The notebooks detect the device automatically and fall back to
CPU with reduced epoch counts, but CPU training is 20–40x slower and the FPS figures measured on
CPU must not be compared with GPU figures from the literature. If no local GPU is available, upload
this folder plus `Clean_Dataset` to Google Colab and run the notebooks there.

Indicative CPU latency measured during pipeline testing (2 vCPU, 512 px, one image at a time):
YOLOv10-n ≈ 89 ms, RT-DETR-l ≈ 748 ms, Faster R-CNN R50-FPNv2 ≈ 2 610 ms. These confirm the
ordering argued in Section 2.4.4 but are not the numbers to report — re-measure on the GPU used for
the final training runs.

## Notes on the data

* The images in `Clean_Dataset` are **512 × 512**, but the VOC XML files still carry the original
  capture dimensions (e.g. 3034 × 1586) and original-pixel boxes. Notebook 00 rescales every box by
  `512 / original_width` and `512 / original_height`; Section 6 of that notebook draws the result so
  the rescaling can be verified visually.
* After rescaling, the defects are roughly **10–25 px** across. This is why notebook 02 replaces
  torchvision's default anchor sizes (32–512 px) with 8–128 px — with the default anchors the RPN
  proposes almost nothing useful and Faster R-CNN trains to near-zero mAP.
* Class ids differ by format on purpose: YOLO uses 0–5, COCO/torchvision uses 1–6 because id 0 is
  reserved for background.

## Outputs

```
dataset/        yolo/ (images, labels, data.yaml), coco/ (instances_*.json), dataset_info.json
runs/           training runs — weights, curves, confusion matrices (per model)
results/        one JSON per model, model_comparison.csv, per_class_comparison.csv,
                comparison_summary.md, figures/*.png (300 dpi, ready for the report)
```
