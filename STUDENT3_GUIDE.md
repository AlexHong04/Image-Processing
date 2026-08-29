# Student 3 — How to get data from Student 2

This guide shows exactly what Student 3 must call to receive the **aligned picture**
from Student 2 (after Student 1 has done the preprocessing), then run defect
detection on it.

> **Flow:** Student 3 → Student 2 → Student 1 → Student 3
>
> - Student 1 preprocesses (3 ways) and returns its result.
> - Student 2 accepts Student 1's returned data (all 3 kinds), does the
>   perspective transform, and returns the aligned result to Student 3.
> - **Student 3 is the only step that saves files.**

---

## 1. Imports (copy this cell first)

Put this at the top of your notebook. It works no matter where your notebook
lives, as long as `image_pipeline.py` is inside the `Image-Processing` folder.

```python
import os, sys, cv2

# Make the shared module importable regardless of the kernel CWD
d = os.path.abspath(os.getcwd())
for _ in range(8):
    if os.path.exists(os.path.join(d, "image_pipeline.py")):
        sys.path.insert(0, d)
        break
    d = os.path.dirname(d)

from image_pipeline import (
    student2_from_student1_array,     # raw image  -> aligned picture (array)
    student2_from_student1_bytes,     # raw bytes  -> aligned picture (array)
    student2_from_student1_video,     # raw video  -> aligned video bytes
    process_from_folder,              # Preprocessed_Dataset -> aligned picture (array)
    save_aligned,                     # Student 3's final save step
)
```

> ⚠️ **Use the `tf_gpu` environment** (Python 3.10, has OpenCV 4.8 + numpy).
> The base conda env currently has a broken numpy.

---

## 2. Get Student 2's aligned picture

Pick the one that matches the input you have.

### A. You have a RAW image (e.g. from `PCB_DATASET/rotation/...`)
```python
raw = cv2.imread("path/to/raw_board.jpg")      # raw BGR image
aligned_pic = student2_from_student1_array(raw)   # S1 preprocess -> S2 align -> picture
```

### B. You have RAW image bytes (upload / API)
```python
aligned_pic = student2_from_student1_bytes(raw_bytes)   # -> aligned picture (array)
```

### C. You have a RAW video (mp4 bytes)
```python
aligned_video = student2_from_student1_video(video_bytes)  # -> aligned video (bytes)
```

### D. You read Student 1's `Preprocessed_Dataset` folder
```python
aligned_pic = process_from_folder("Missing_hole_rotation", "01_missing_hole_01.jpg")
```

> Each call returns **in memory** — nothing is written to disk by Student 1 or 2.
> `aligned_pic` is a numpy array (BGR) ready for your detector.
> If the board can't be aligned, the function returns `None` — check before using.

---

## 3. Run your defect detection

```python
if aligned_pic is not None:
    defects = your_detector(aligned_pic)   # YOLO / Faster R-CNN / RT-DETR ...
    print(defects)
```

---

## 4. Save results (final step — only Student 3 saves)

```python
if aligned_pic is not None:
    save_aligned(aligned_pic, "outputs/aligned_01.jpg")
```

Or with plain OpenCV:
```python
if aligned_pic is not None:
    cv2.imwrite("outputs/aligned_01.jpg", aligned_pic)
```

---

## Quick reference

| Student 3 has | Call | Returns |
|---|---|---|
| Raw image (array) | `student2_from_student1_array(raw)` | aligned array (picture) |
| Raw image (bytes) | `student2_from_student1_bytes(bytes)` | aligned array (picture) |
| Raw video (bytes) | `student2_from_student1_video(bytes)` | aligned video bytes |
| `Preprocessed_Dataset` path | `process_from_folder(cls, filename)` | aligned array (picture) |
| Save output (final step) | `save_aligned(array, path)` | writes a file |

## Notes
- `student2_from_student1_*` calls Student 1's **actual** functions from
  `Student1-Image Acquisition & Pre-processing/ImagePreprocessing.ipynb`
  (`preprocess_image`, `process_image_from_bytes`, `process_full_video_backend`)
  and then applies Student 2's perspective transform.
- Student 1 and Student 2 never write files. Only Student 3 does.
- Aligned output preserves the board's aspect ratio (longest side 640).
