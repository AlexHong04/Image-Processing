#!/usr/bin/env python3
"""
image_pipeline.py
=================
Shared, in-memory PCB inspection pipeline for the Image-Processing project.

This module is the SINGLE SOURCE OF TRUTH for the Student 1 -> Student 2 chain
so that Student 3 can call one import and get results back in memory.

IMPORTANT DESIGN RULE
---------------------
Every function here returns its result IN MEMORY (numpy arrays / bytes) and
performs NO permanent file writes. Writing results to disk is intentionally
deferred to Student 3's final output step.

Flow (each step returns its result to the caller):
    Student 1 : preprocess_image(img)           array  -> preprocessed array
                preprocess_from_bytes(bytes)    bytes  -> preprocessed array
                preprocess_video_bytes(bytes)   video  -> preprocessed video bytes
    Student 2 : align_image(img)                array  -> aligned array
    Wired     : process_image_array / process_image_bytes / process_video_bytes
                (Student 1 -> Student 2, all in-memory)
    Folder    : process_from_folder(cls, filename) reads Preprocessed_Dataset
                (Student 1's output folder) and returns the aligned array.

Usage from Student 2 / Student 3 notebooks:
    from image_pipeline import (
        process_image_array, process_image_bytes, process_video_bytes,
        process_from_folder, align_image, preprocess_image,
    )
"""

import contextlib
import io
import json
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent

ALIGN_TARGET = 640                 # longest side of the aligned output
PREPROCESSED_DIR = ROOT / "Preprocessed_Dataset"   # Student 1 output folder
CLEAN_DIR = ROOT / "Clean_Dataset"                 # Student 1 input folder
RAW_ROTATION_DIR = ROOT / "PCB_DATASET" / "rotation"

_JPG_QUALITY = 95                  # JPEG encode quality for byte outputs
_VIDEO_FOURCC = "mp4v"             # MP4 codec for video outputs


# --------------------------------------------------------------------------- #
# Student 1 : preprocessing (in-memory)
# --------------------------------------------------------------------------- #
def preprocess_image(img):
    """
    Student-1 core preprocessing: Colour Normalisation -> Gaussian Filter
    -> CLAHE on the L channel of LAB.

    Args:
        img: OpenCV BGR image (numpy array).

    Returns:
        Processed BGR image (numpy array), or None if `img` is empty.
    """
    if img is None:
        return None

    # 1. Colour Normalisation (min-max stretch to full 0-255 range)
    normalised = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

    # 2. Gaussian Filter
    gaussian = cv2.GaussianBlur(normalised, (5, 5), 0)

    # 3. Convert to LAB colour space for CLAHE
    lab = cv2.cvtColor(gaussian, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to the L-channel only
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)

    # 4. Merge channels and convert back to BGR
    lab_clahe = cv2.merge((l_clahe, a, b))
    processed = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    return processed


def preprocess_from_bytes(image_bytes):
    """
    Student-1 (output kind 1 - bytes): raw image bytes -> preprocessed BGR array.

    Args:
        image_bytes: raw encoded image bytes (JPEG/PNG/...).

    Returns:
        Preprocessed BGR array, or None on failure.
    """
    if not image_bytes:
        return None
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return preprocess_image(img)


def preprocess_video_bytes(video_bytes):
    """
    Student-1 (output kind 3 - video): video bytes -> processed video bytes.

    Each frame is passed through `preprocess_image`. A temporary file is used
    internally only because OpenCV's VideoWriter needs a path; the result is
    returned as bytes and the temp files are deleted (nothing persists).

    Args:
        video_bytes: bytes of an uploaded .mp4 video.

    Returns:
        Processed video bytes, or None on failure.
    """
    if not video_bytes:
        return None

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
        tmp_in.write(video_bytes)
        tmp_in_path = tmp_in.name

    out_path = None
    try:
        cap = cv2.VideoCapture(tmp_in_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        out_path = tempfile.mktemp(suffix="_processed.mp4")
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*_VIDEO_FOURCC),
                              fps, (width, height))

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            processed = preprocess_image(frame)
            if processed is not None:
                out.write(processed)

        cap.release()
        out.release()

        with open(out_path, "rb") as f:
            return f.read()
    except Exception as exc:  # noqa: BLE001 - API-style catch-all
        print(f"Error processing video: {exc}")
        return None
    finally:
        if os.path.exists(tmp_in_path):
            os.remove(tmp_in_path)
        if out_path and os.path.exists(out_path):
            os.remove(out_path)


# --------------------------------------------------------------------------- #
# Student 2 : alignment & calibration (in-memory)
# --------------------------------------------------------------------------- #
def board_corners_threshold(img):
    """
    Detect the 4 corner points of the PCB board using Otsu thresholding.
    The board is separated from the background by brightness, so thresholding
    is far more reliable than Canny (which fragments the board outline and
    locks onto tiny internal features).

    Returns (corners_4x2 float32, board coverage) or (None, coverage).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu threshold (invert: board = foreground / white)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological close to bridge small gaps in the board outline
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0.0

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest) / (img.shape[0] * img.shape[1])
    perim = cv2.arcLength(largest, True)

    # Try a few epsilon factors to get exactly 4 corners
    for eps_factor in (0.02, 0.01, 0.03, 0.04):
        approx = cv2.approxPolyDP(largest, eps_factor * perim, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype("float32"), area

    return None, area


def order_points(points):
    """
    Orders 4 corner points as: top-left, top-right, bottom-right, bottom-left.
    """
    sum_points = points.sum(axis=1)
    diff_points = np.diff(points, axis=1).reshape(4)

    ordered = np.zeros((4, 2), dtype="float32")
    ordered[0] = points[np.argmin(sum_points)]   # Top-left
    ordered[2] = points[np.argmax(sum_points)]   # Bottom-right
    ordered[1] = points[np.argmin(diff_points)]  # Top-right
    ordered[3] = points[np.argmax(diff_points)]  # Bottom-left

    return ordered


def compute_destination(ordered, target_longest=ALIGN_TARGET):
    """
    Given 4 ordered corners (TL, TR, BR, BL), return (dst_points, width, height)
    for the warped output. The board's aspect ratio is preserved by scaling the
    longest side to `target_longest`, which avoids squeezing the PCB into a square.
    """
    tl, tr, br, bl = ordered
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)

    out_w = max(width_top, width_bottom)
    out_h = max(height_left, height_right)

    if out_w <= 0 or out_h <= 0:
        return None, 0, 0

    scale = target_longest / max(out_w, out_h)
    out_w = int(round(out_w * scale))
    out_h = int(round(out_h * scale))

    dst = np.float32([
        [0, 0],
        [out_w, 0],
        [out_w, out_h],
        [0, out_h]
    ])
    return dst, out_w, out_h


def align_image(img):
    """
    Detects the PCB boundary in `img` and warps it to a top-down view,
    preserving the board's aspect ratio (longest side = ALIGN_TARGET).

    Args:
        img: OpenCV BGR image (numpy array).

    Returns:
        Aligned BGR image (numpy array), or None if the board boundary
        could not be resolved to a clean quadrilateral.
    """
    if img is None:
        return None

    corners, _ = board_corners_threshold(img)
    if corners is None:
        return None

    ordered = order_points(corners)
    dst, out_w, out_h = compute_destination(ordered)
    if dst is None:
        return None

    matrix = cv2.getPerspectiveTransform(ordered, dst)
    return cv2.warpPerspective(img, matrix, (out_w, out_h))


def quick_corner_check(img):
    """True if `img` contains a clean 4-corner PCB boundary."""
    corners, _ = board_corners_threshold(img)
    return corners is not None


# --------------------------------------------------------------------------- #
# Wired pipeline : Student 1 -> Student 2  (all in-memory)
# --------------------------------------------------------------------------- #
def process_image_array(img, do_preprocess=False):
    """
    Wired output kind 1 (array): image array -> aligned array.

    Args:
        img: OpenCV BGR image (numpy array).
        do_preprocess: if True, first runs Student 1's preprocessing
                       (use when `img` is a RAW image, e.g. from
                       PCB_DATASET/rotation). Default False = `img` is
                       already preprocessed (e.g. from Preprocessed_Dataset).

    Returns:
        Aligned BGR array, or None if the board was not found.
    """
    if img is None:
        return None
    if do_preprocess:
        img = preprocess_image(img)
        if img is None:
            return None
    return align_image(img)


def process_image_bytes(image_bytes, do_preprocess=False):
    """
    Wired output kind 2 (bytes): raw image bytes -> aligned JPEG bytes.

    Matches Student 1's `process_image_from_bytes` API pattern but adds
    Student 2's alignment, so the frontend/Student 3 gets aligned bytes back.

    Returns:
        Aligned JPEG bytes, or None on failure.
    """
    if not image_bytes:
        return None
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    aligned = process_image_array(img, do_preprocess=do_preprocess)
    if aligned is None:
        return None
    ok, encoded = cv2.imencode(".jpg", aligned, [cv2.IMWRITE_JPEG_QUALITY, _JPG_QUALITY])
    if not ok:
        return None
    return encoded.tobytes()


def process_video_bytes(video_bytes, do_preprocess=False):
    """
    Wired output kind 3 (video): video bytes -> aligned video bytes.

    Every frame is passed through Student 1's preprocessing (optional) and
    Student 2's alignment, then re-encoded. Returns the processed video as
    bytes (in memory); temp files are cleaned up automatically.

    Returns:
        Aligned video bytes, or None on failure.
    """
    if not video_bytes:
        return None

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
        tmp_in.write(video_bytes)
        tmp_in_path = tmp_in.name

    out_path = None
    try:
        cap = cv2.VideoCapture(tmp_in_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        out_path = tempfile.mktemp(suffix="_aligned.mp4")
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*_VIDEO_FOURCC),
                              fps, (width, height))

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            aligned = process_image_array(frame, do_preprocess=do_preprocess)
            if aligned is not None:
                out.write(aligned)

        cap.release()
        out.release()

        with open(out_path, "rb") as f:
            return f.read()
    except Exception as exc:  # noqa: BLE001 - API-style catch-all
        print(f"Error processing video: {exc}")
        return None
    finally:
        if os.path.exists(tmp_in_path):
            os.remove(tmp_in_path)
        if out_path and os.path.exists(out_path):
            os.remove(out_path)


# --------------------------------------------------------------------------- #
# Student 2 calls Student 1's ACTUAL notebook functions from the start
# --------------------------------------------------------------------------- #
STUDENT1_NOTEBOOK = (
    ROOT / "Student1-Image Acquisition & Pre-processing" / "ImagePreprocessing.ipynb"
)
_STUDENT1_FN_NAMES = ("preprocess_image", "process_image_from_bytes", "process_full_video_backend")
_STUDENT1_CACHE = None


def load_student1_functions():
    """
    Dynamically load Student 1's three functions directly from their notebook
    (single source of truth = Student 1's notebook). Only the function-definition
    cells are executed, so no demo/batch side effects. Cached after first call.

    Returns a dict: {name: function} for
    'preprocess_image', 'process_image_from_bytes', 'process_full_video_backend'.
    """
    global _STUDENT1_CACHE
    if _STUDENT1_CACHE is not None:
        return _STUDENT1_CACHE

    if not STUDENT1_NOTEBOOK.exists():
        raise FileNotFoundError(f"Student 1 notebook not found: {STUDENT1_NOTEBOOK}")

    # Namespace with the dependencies Student 1's functions need.
    ns = {"cv2": cv2, "np": np, "os": os, "tempfile": tempfile}

    with open(STUDENT1_NOTEBOOK, encoding="utf-8") as f:
        nb = json.load(f)

    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell.get("source", []))
        # Only exec cells that DEFINE one of Student 1's three functions.
        if any(("def " + name + "(") in src for name in _STUDENT1_FN_NAMES):
            # Suppress any prints in Student 1's cells so the notebook output stays clean.
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                exec(compile(src, str(STUDENT1_NOTEBOOK), "exec"), ns)

    _STUDENT1_CACHE = {name: ns[name] for name in _STUDENT1_FN_NAMES if name in ns}
    return _STUDENT1_CACHE


def student2_from_student1_array(raw_img):
    """
    Student 2 calls Student 1's `preprocess_image` from the start, then aligns,
    and returns the aligned ARRAY to Student 3.

    Args:
        raw_img: raw BGR image (numpy array) to be preprocessed by Student 1.

    Returns:
        aligned BGR array (for Student 3), or None on failure.
    """
    s1 = load_student1_functions()
    processed = s1["preprocess_image"](raw_img)      # Student 1 -> preprocessed array
    if processed is None:
        return None
    return align_image(processed)                    # Student 2 -> aligned array


def student2_from_student1_bytes(image_bytes):
    """
    Student 2 calls Student 1's `process_image_from_bytes` from the start, then
    aligns the returned image, and returns the aligned ARRAY to Student 3.

    Args:
        image_bytes: raw image bytes (as Student 1's bytes input expects).

    Returns:
        aligned BGR array (for Student 3), or None on failure.
    """
    s1 = load_student1_functions()
    processed = s1["process_image_from_bytes"](image_bytes)   # Student 1 -> JPEG bytes
    if processed is None:
        return None
    img = cv2.imdecode(np.frombuffer(processed, np.uint8), cv2.IMREAD_COLOR)
    return align_image(img)                                    # Student 2 -> aligned array


def student2_from_student1_video(video_bytes):
    """
    Student 2 calls Student 1's `process_full_video_backend` from the start,
    aligns every frame of the returned video, and returns aligned video BYTES
    to Student 3 (temp files cleaned up).

    Args:
        video_bytes: raw mp4 video bytes.

    Returns:
        aligned mp4 bytes (for Student 3), or None on failure.
    """
    s1 = load_student1_functions()
    processed_path = s1["process_full_video_backend"](video_bytes)  # Student 1 -> temp path
    if not processed_path or not os.path.exists(processed_path):
        return None
    try:
        with open(processed_path, "rb") as f:
            processed_video = f.read()
    finally:
        if os.path.exists(processed_path):
            os.remove(processed_path)
    return process_video_bytes(processed_video, do_preprocess=False)  # Student 2 -> aligned bytes


# --------------------------------------------------------------------------- #
# Folder source : read Student 1's Preprocessed_Dataset and align (in-memory)
# --------------------------------------------------------------------------- #
def _read_folder_image(folder, cls, filename):
    path = os.path.join(str(folder), cls, filename)
    if not os.path.exists(path):
        return None
    return cv2.imread(path)


def read_preprocessed(cls, filename):
    """Read a Student-1 preprocessed image (array) from Preprocessed_Dataset."""
    return _read_folder_image(PREPROCESSED_DIR, cls, filename)


def process_from_folder(cls, filename, do_preprocess=False):
    """
    Full folder flow used by Student 3:
    read Preprocessed_Dataset/<cls>/<filename> -> align -> return array.

    Returns aligned BGR array (in memory), or None if missing/un-alignable.
    """
    img = read_preprocessed(cls, filename)
    return process_image_array(img, do_preprocess=do_preprocess)


# --------------------------------------------------------------------------- #
# Convenience for Student 3's FINAL output step (the only place that saves)
# --------------------------------------------------------------------------- #
def save_aligned(array, save_path):
    """Write an aligned array to `save_path`. Intended to be called by Student 3."""
    if array is None:
        return False
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    return cv2.imwrite(save_path, array)
