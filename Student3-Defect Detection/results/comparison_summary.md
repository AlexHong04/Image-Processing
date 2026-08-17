# Defect detection - model comparison summary

Dataset: Clean_Dataset (6 PCB defect classes), evaluated on the held-out test split.

| Model | Type | Params (M) | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | FPS | Device |
|---|---|---|---|---|---|---|---|---|
| YOLOv10 | single-stage (NMS-free) | 2.78 | 0.566 | 0.249 | 0.565 | 0.527 | 19.2 | CPU |
| Faster R-CNN | two-stage (RPN + RoIAlign, with NMS) | 43.28 | 0.467 | 0.164 | 0.714 | 0.164 | 0.4 | CPU |
| RT-DETR | end-to-end transformer (NMS-free) | 32.97 | 0.669 | 0.293 | 0.701 | 0.637 | 2.1 | CPU |

## Findings

- Highest accuracy: **RT-DETR** with mAP@0.5:0.95 = 0.293 and mAP@0.5 = 0.669.
- Highest throughput: **YOLOv10** at 19.2 FPS (52.1 ms per image).
- Accuracy spread across the three architectures: 0.129 mAP@0.5:0.95.
- Hardest defect class overall: spurious_copper.
- Easiest defect class overall: missing_hole.

Figures for the report are in `results/figures/`.