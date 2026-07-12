<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Image Processing — PCB Defect Inspection</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Segoe UI', Tahoma, Geneva, sans-serif;
    font-size: 14px; color: #2D2D2D; background: #ffffff;
    padding: 40px; max-width: 960px; margin: 0 auto;
  }
  .hero { text-align: center; margin-bottom: 8px; }
  .hero-title { font-size: 28px; font-weight: 700; color: #1E4D8C; margin-bottom: 4px; }
  .hero-sub   { font-size: 20px; color: #2E75B6; margin-bottom: 16px; }
  .badge-bar { display: grid; grid-template-columns: 1fr 1fr 1fr; margin-bottom: 20px; border-radius: 4px; overflow: hidden; }
  .badge { padding: 8px 12px; text-align: center; font-weight: 700; font-size: 12.5px; color: #fff; }
  .badge-navy  { background: #1E4D8C; }
  .badge-blue  { background: #2E75B6; }
  .badge-green { background: #4CAF50; }
  .divider { border: none; border-bottom: 1px solid #D0D8E8; margin: 20px 0; }
  h1 { font-size: 17.5px; font-weight: 700; color: #1E4D8C; margin: 28px 0 10px; }
  h2 { font-size: 15px;   font-weight: 700; color: #2E75B6; margin: 20px 0 8px; }
  p  { font-size: 13.75px; color: #2D2D2D; margin-bottom: 8px; line-height: 1.5; }
  .code-block {
    background: #F0F4F8;
    border-top: 1px solid #C5D5E8; border-right: 1px solid #C5D5E8;
    border-bottom: 1px solid #C5D5E8; border-left: 5px solid #2E75B6;
    padding: 10px 14px; margin: 8px 0;
    font-family: 'Courier New', Courier, monospace;
    font-size: 11.25px; color: #1A1A1A; line-height: 1.9; white-space: pre;
  }
  .tbl { width: 100%; border-collapse: collapse; margin: 8px 0; font-size: 12.5px; }
  .tbl th { background: #1E4D8C; color: #fff; font-weight: 700; padding: 7px 10px; text-align: left; border: 1px solid #CCCCCC; }
  .tbl td { padding: 7px 10px; border: 1px solid #CCCCCC; vertical-align: top; line-height: 1.4; }
  .tbl tr:nth-child(odd)  td { background: #F0F4F8; }
  .tbl tr:nth-child(even) td { background: #ffffff; }
  .tbl td code { font-family: 'Courier New', Courier, monospace; font-size: 11px; color: #1A1A1A; }
  .tbl-prereq { width: 100%; border-collapse: collapse; margin: 6px 0; font-size: 12.5px; border: 1.5px solid #1E4D8C; }
  .tbl-prereq td { padding: 7px 10px; vertical-align: middle; border: none; }
  .tbl-prereq .label { background: #EEF4FB; font-weight: 700; color: #1E4D8C; width: 160px; text-align: center; }
  .tbl-prereq .desc  { background: #ffffff; color: #2D2D2D; }
  .tbl-install { width: 100%; border-collapse: collapse; margin: 6px 0; font-size: 12.5px; border: 1.5px solid #217346; }
  .tbl-install td { padding: 7px 10px; vertical-align: middle; border: none; }
  .tbl-install .label { background: #E8F5E9; font-weight: 700; color: #217346; width: 160px; text-align: center; }
  .tbl-install .desc  { background: #ffffff; color: #2D2D2D; }
  .tip-box  { display: grid; grid-template-columns: 110px 1fr; margin: 8px 0; border: 1.5px solid #E65100; font-size: 12.5px; }
  .tip-box  .tip-label  { background: #FFF3E0; font-weight: 700; color: #E65100; padding: 8px 10px; text-align: center; display: flex; align-items: center; justify-content: center; }
  .tip-box  .tip-body   { background: #ffffff; padding: 8px 12px; color: #2D2D2D; line-height: 1.5; }
  .note-box { display: grid; grid-template-columns: 110px 1fr; margin: 8px 0; border: 1.5px solid #2E75B6; font-size: 12.5px; }
  .note-box .note-label { background: #EEF4FB; font-weight: 700; color: #2E75B6; padding: 8px 10px; text-align: center; display: flex; align-items: center; justify-content: center; }
  .note-box .note-body  { background: #ffffff; padding: 8px 12px; color: #2D2D2D; line-height: 1.5; }
  .tbl-flow .step-num { font-weight: 700; color: #1E4D8C; text-align: center; width: 50px; font-size: 13px; }
  .tbl-flow tr:nth-child(odd)  td { background: #EEF4FB; }
  .tbl-flow tr:nth-child(even) td { background: #ffffff; }
  .tbl-flow td.note { color: #555555; font-size: 11.75px; }
  .footer-bar {
    margin-top: 24px; border: 1.5px solid #1E4D8C; border-top: 4px solid #1E4D8C;
    background: #EEF4FB; padding: 12px 16px; text-align: center;
    font-size: 11.5px; font-weight: 700; color: #1E4D8C;
  }
</style>
</head>
<body>

<!-- HERO -->
<div class="hero">
  <div class="hero-title">🔬  Image Processing</div>
  <div class="hero-sub">PCB Defect Inspection System — Module 1</div>
</div>

<div class="badge-bar">
  <div class="badge badge-navy">🐍  Python 3.x + OpenCV</div>
  <div class="badge badge-blue">📊  Matplotlib + NumPy</div>
  <div class="badge badge-green">📓  Jupyter Notebook</div>
</div>

<hr class="divider">

<!-- OVERVIEW -->
<h1>📋  Project Overview</h1>
<p>This module handles <strong>Image Acquisition &amp; Pre-processing</strong> for automated PCB defect detection. Raw PCB images are cleaned, resized, colour-normalised, and contrast-enhanced before being passed to downstream detection modules.</p>

<table class="tbl">
  <tr><th>Attribute</th><th>Details</th></tr>
  <tr><td>Module</td><td>Student 1 — Image Acquisition &amp; Pre-processing</td></tr>
  <tr><td>Cleaning Algorithms</td><td>Corrupt image removal, Bicubic Interpolation resize (512×512)</td></tr>
  <tr><td>Preprocessing Algorithms</td><td>Min-Max Normalisation, RGB→LAB Colour Conversion, CLAHE</td></tr>
  <tr><td>Input Dataset</td><td><code>PCB_DATASET/images/</code></td></tr>
  <tr><td>Output Dataset</td><td><code>Preprocessed_Dataset/</code></td></tr>
</table>

<hr class="divider">

<!-- REPO STRUCTURE -->
<h1>📁  Repository Structure</h1>
<div class="code-block">Image Processing/
│
├── PCB_DATASET/                          ← Original PCB defect images (input)
│   └── images/
│       ├── &lt;class_1&gt;/
│       └── &lt;class_n&gt;/
│
├── Clean_Dataset/                        ← Resized &amp; validated images (512×512)
│   ├── &lt;class_1&gt;/
│   └── &lt;class_n&gt;/
│
├── Preprocessed_Dataset/                 ← Normalised &amp; contrast-enhanced output
│   ├── &lt;class_1&gt;/
│   └── &lt;class_n&gt;/
│
└── Student1-Image Acquisition &amp; Pre-proc/
    ├── DataCleanning.ipynb               ← Step 1: Validation, resize, class summary
    └── ImagePreprocessing.ipynb          ← Step 2: Normalisation, LAB, CLAHE</div>

<hr class="divider">

<!-- PREREQUISITES -->
<h1>📦  Prerequisites</h1>
<p>Ensure the following are installed before running any notebook:</p>

<table class="tbl-prereq"><tr><td class="label">Python</td><td class="desc">Version 3.8 or later — https://www.python.org/downloads</td></tr></table>
<table class="tbl-prereq"><tr><td class="label">Jupyter Notebook</td><td class="desc">Via Anaconda or <code>pip install notebook</code></td></tr></table>
<table class="tbl-prereq"><tr><td class="label">OpenCV</td><td class="desc">Image I/O, resize, colour conversion, CLAHE</td></tr></table>
<table class="tbl-prereq"><tr><td class="label">Matplotlib</td><td class="desc">Visualisation of sample and before/after results</td></tr></table>
<table class="tbl-prereq"><tr><td class="label">NumPy</td><td class="desc">Numerical array operations underlying all transformations</td></tr></table>

<hr class="divider">

<!-- INSTALLATION -->
<h1>⚙️  Installation</h1>

<h2>💻  Install via pip</h2>
<div class="code-block">python -m pip install opencv-python
python -m pip install matplotlib numpy</div>

<h2>💻  Install all at once</h2>
<div class="code-block">python -m pip install opencv-python matplotlib numpy</div>

<div class="tip-box">
  <div class="tip-label">💡 Tip</div>
  <div class="tip-body">Use a virtual environment or Anaconda environment to avoid dependency conflicts with other projects.</div>
</div>

<hr class="divider">

<!-- NOTEBOOKS -->
<h1>📓  Notebooks</h1>

<h2>🧹  DataCleanning.ipynb — Data Cleaning</h2>
<p>Validates, filters, and resizes all raw PCB images from <code>PCB_DATASET/images/</code> into <code>Clean_Dataset/</code>.</p>

<table class="tbl">
  <tr><th>Section</th><th>What It Does</th></tr>
  <tr><td>Set Dataset Path</td><td>Defines input (<code>PCB_DATASET/images</code>) and output (<code>Clean_Dataset</code>) folders</td></tr>
  <tr><td>Load Dataset</td><td>Scans all class subfolders; counts total <code>.jpg/.jpeg/.png</code> images per class</td></tr>
  <tr><td>Check Image</td><td>Attempts <code>cv2.imread()</code> on every file; separates valid vs corrupted images</td></tr>
  <tr><td>Display Sample Images</td><td>Randomly displays 5 valid images with class labels for visual inspection</td></tr>
  <tr><td>Image Cleaning</td><td>Resizes every valid image to <strong>512×512</strong> using Bicubic Interpolation; saves per-class to <code>Clean_Dataset/</code></td></tr>
  <tr><td>Display Cleaned Images</td><td>Randomly displays 5 cleaned images from a randomly selected class</td></tr>
  <tr><td>Dataset Summary</td><td>Prints per-class count and totals for original, valid, corrupted, and processed images</td></tr>
</table>

<h2>🖼️  ImagePreprocessing.ipynb — Image Preprocessing</h2>
<p>Applies the full pre-processing pipeline to images in <code>Clean_Dataset/</code> and saves results to <code>Preprocessed_Dataset/</code>.</p>

<table class="tbl">
  <tr><th>Algorithm</th><th>OpenCV Function</th><th>Purpose</th></tr>
  <tr><td>Min-Max Normalisation</td><td><code>cv2.normalize(..., cv2.NORM_MINMAX)</code></td><td>Scales pixel values to [0, 255] to reduce brightness variation across images</td></tr>
  <tr><td>RGB → LAB Conversion</td><td><code>cv2.cvtColor(..., cv2.COLOR_BGR2LAB)</code></td><td>Separates luminance (L) from colour channels (A, B) for independent processing</td></tr>
  <tr><td>CLAHE</td><td><code>cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))</code></td><td>Enhances local contrast on the L channel to improve defect visibility</td></tr>
</table>

<div class="note-box">
  <div class="note-label">📌 Note</div>
  <div class="note-body">CLAHE is applied <strong>only to the L (luminance) channel</strong> after LAB conversion. The A and B colour channels are kept unchanged, then the image is converted back to BGR before saving.</div>
</div>

<hr class="divider">

<!-- PIPELINE -->
<h1>🔄  Preprocessing Pipeline</h1>
<div class="code-block">Input (Clean_Dataset/)
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
    └── cv2.imwrite()            ← Save to Preprocessed_Dataset/</div>

<hr class="divider">

<!-- HOW TO RUN -->
<h1>🚀  How to Run</h1>
<p>Follow this order when running the notebooks:</p>

<table class="tbl tbl-flow">
  <tr><th style="width:50px">Step</th><th>Action</th><th class="note">Notes</th></tr>
  <tr><td class="step-num">1</td><td><strong>📂  Place raw images</strong></td><td class="note">Copy PCB defect images into <code>PCB_DATASET/images/&lt;class&gt;/</code></td></tr>
  <tr><td class="step-num">2</td><td><strong>🧹  Run DataCleanning.ipynb</strong></td><td class="note">Run all cells — outputs go to <code>Clean_Dataset/</code></td></tr>
  <tr><td class="step-num">3</td><td><strong>🖼️  Run ImagePreprocessing.ipynb</strong></td><td class="note">Run all cells — outputs go to <code>Preprocessed_Dataset/</code></td></tr>
  <tr><td class="step-num">4</td><td><strong>✅  Verify output</strong></td><td class="note">Check before/after visualisation and dataset summary printout</td></tr>
</table>

<div class="tip-box">
  <div class="tip-label">⚠️ Order</div>
  <div class="tip-body">Always run <strong>DataCleanning.ipynb first</strong>. The preprocessing notebook reads from <code>Clean_Dataset/</code>, which is only populated after the cleaning step completes.</div>
</div>

<hr class="divider">

<!-- QUICK REFERENCE -->
<h1>⚡  Quick Reference</h1>

<h2>💻  Install Dependencies</h2>
<div class="code-block">python -m pip install opencv-python
python -m pip install matplotlib numpy</div>

<h2>📓  Notebook Run Order</h2>
<div class="code-block">1. DataCleanning.ipynb       →  Clean_Dataset/          (512×512 resized)
2. ImagePreprocessing.ipynb  →  Preprocessed_Dataset/   (normalised + CLAHE)</div>

<table class="tbl-install"><tr><td class="label">📈 Final Output</td><td class="desc"><code>Preprocessed_Dataset/</code> — BGR images, Min-Max normalised, LAB-CLAHE enhanced, ready for Module 2</td></tr></table>

<!-- FOOTER -->
<div class="footer-bar">
  🔬  Image Processing · PCB Defect Inspection System  —  Student 1: Image Acquisition &amp; Pre-processing
</div>

</body>
</html>
