# 🔬 Image Processing
## PCB Defect Inspection System — Module 1

![Python](https://img.shields.io/badge/Python-3.8+-blue) ![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green) ![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)

---

## 📋 Project Overview

This module handles **Image Acquisition & Pre-processing** for automated PCB defect detection. Raw PCB images are cleaned, resized, colour-normalised, and contrast-enhanced before being passed to downstream detection modules.

| Attribute | Details |
|-----------|---------|
| Module | Student 1 — Image Acquisition & Pre-processing |
| Cleaning Algorithms | Corrupt image removal, Bicubic Interpolation resize (512×512) |
| Preprocessing Algorithms | Min-Max Normalisation, RGB→LAB Colour Conversion, CLAHE |
| Input Dataset | `PCB_DATASET/images/` |
| Output Dataset | `Preprocessed_Dataset/` |

---

## 📁 Repository Structure

```
Image Processing/
│
├── PCB_DATASET/                          ← Original PCB defect images (input)
│   └── images/
│       ├── <class_1>/
│       └── <class_n>/
│
├── Clean_Dataset/                        ← Resized & validated images (512×512)
│   ├── <class_1>/
│   └── <class_n>/
│
├── Preprocessed_Dataset/                 ← Normalised & contrast-enhanced output
│   ├── <class_1>/
│   └── <class_n>/
│
└── Student1-Image Acquisition & Pre-proc/
    ├── DataCleanning.ipynb               ← Step 1: Validation, resize, class summary
    └── ImagePreprocessing.ipynb          ← Step 2: Normalisation, LAB, CLAHE
```

---

## 📦 Prerequisites

Ensure the following are installed before running any notebook:

| Tool | Details |
|------|---------|
| Python | Version 3.8 or later — https://www.python.org/downloads |
| Jupyter Notebook | Via Anaconda or `pip install notebook` |
| OpenCV | Image I/O, resize, colour conversion, CLAHE |
| Matplotlib | Visualisation of sample and before/after results |
| NumPy | Numerical array operations underlying all transformations |

---

## ⚙️ Installation

### Install via pip

```bash
python -m pip install opencv-python
python -m pip install matplotlib numpy
```

### Install all at once

```bash
python -m pip install opencv-python matplotlib numpy
```

> 💡 **Tip:** Use a virtual environment or Anaconda environment to avoid dependency conflicts with other projects.

---

## 📓 Notebooks

### 🧹 DataCleanning.ipynb — Data Cleaning

Validates, filters, and resizes all raw PCB images from `PCB_DATASET/images/` into `Clean_Dataset/`.

| Section | What It Does |
|---------|-------------|
| Set Dataset Path | Defines input (`PCB_DATASET/images`) and output (`Clean_Dataset`) folders |
| Load Dataset | Scans all class subfolders; counts total `.jpg/.jpeg/.png` images per class |
| Check Image | Attempts `cv2.imread()` on every file; separates valid vs corrupted images |
| Display Sample Images | Randomly displays 5 valid images with class labels for visual inspection |
| Image Cleaning | Resizes every valid image to **512×512** using Bicubic Interpolation; saves per-class to `Clean_Dataset/` |
| Display Cleaned Images | Randomly displays 5 cleaned images from a randomly selected class |
| Dataset Summary | Prints per-class count and totals for original, valid, corrupted, and processed images |

---

### 🖼️ ImagePreprocessing.ipynb — Image Preprocessing

Applies the full pre-processing pipeline to images in `Clean_Dataset/` and saves results to `Preprocessed_Dataset/`.

| Algorithm | OpenCV Function | Purpose |
|-----------|----------------|---------|
| Min-Max Normalisation | `cv2.normalize(..., cv2.NORM_MINMAX)` | Scales pixel values to [0, 255] to reduce brightness variation across images |
| RGB → LAB Conversion | `cv2.cvtColor(..., cv2.COLOR_BGR2LAB)` | Separates luminance (L) from colour channels (A, B) for independent processing |
| CLAHE | `cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))` | Enhances local contrast on the L channel to improve defect visibility |

> 📌 **Note:** CLAHE is applied **only to the L (luminance) channel** after LAB conversion. The A and B colour channels are kept unchanged, then the image is converted back to BGR before saving.

---

## 🔄 Preprocessing Pipeline

```
Input (Clean_Dataset/)
    │
    ├── cv2.normalize()          ← Min-Max Normalisation  [0–255]
    │
    ├── cv2.COLOR_BGR2LAB        ← Convert to LAB colour space
    │
    ├── CLAHE on L channel       ← Contrast enhancement (clipLimit=2.0, tile=8×8)
    │
    ├── cv2.merge(L, A, B)       ← Recombine channels
    │
    ├── cv2.COLOR_LAB2BGR        ← Convert back to BGR
    │
    └── cv2.imwrite()            ← Save to Preprocessed_Dataset/
```

---

## 🚀 How to Run

Follow this order when running the notebooks:

| Step | Action | Notes |
|------|--------|-------|
| 1 | 📂 Place raw images | Copy PCB defect images into `PCB_DATASET/images/<class>/` |
| 2 | 🧹 Run DataCleanning.ipynb | Run all cells — outputs go to `Clean_Dataset/` |
| 3 | 🖼️ Run ImagePreprocessing.ipynb | Run all cells — outputs go to `Preprocessed_Dataset/` |
| 4 | ✅ Verify output | Check before/after visualisation and dataset summary printout |

> ⚠️ **Order matters:** Always run **DataCleanning.ipynb first**. The preprocessing notebook reads from `Clean_Dataset/`, which is only populated after the cleaning step completes.

---

## ⚡ Quick Reference

### Install Dependencies

```bash
python -m pip install opencv-python
python -m pip install matplotlib numpy
```

### Notebook Run Order

```
1. DataCleanning.ipynb       →  Clean_Dataset/          (512×512 resized)
2. ImagePreprocessing.ipynb  →  Preprocessed_Dataset/   (normalised + CLAHE)
```

---

*🔬 Image Processing · PCB Defect Inspection System — Student 1: Image Acquisition & Pre-processing*