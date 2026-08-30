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

import ast
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

ALIGN_TARGET = 1280                # longest side of the aligned output.
#   640 shrinks a ~70 px PCB defect to ~15 px, which is below what a detector
#   can reliably find. 1280 keeps defects at ~30 px. The detector is TRAINED at
#   this same value - change it here and the training set must be rebuilt.
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


def align_image_with_matrix(img, target_longest=None):
    """
    Same as `align_image` but ALSO returns the homography and the output size.

    The homography is what lets ground-truth boxes (or any other coordinate) be
    carried from the input image into the aligned image, which is required to
    build a training set that matches what the detector sees at run time.

    Returns:
        (aligned_bgr, H_3x3, (out_w, out_h)) or (None, None, None) on failure.
    """
    if img is None:
        return None, None, None

    corners, _ = board_corners_threshold(img)
    if corners is None:
        return None, None, None

    ordered = order_points(corners)
    dst, out_w, out_h = compute_destination(ordered, target_longest or ALIGN_TARGET)
    if dst is None:
        return None, None, None

    matrix = cv2.getPerspectiveTransform(ordered, dst)
    return cv2.warpPerspective(img, matrix, (out_w, out_h)), matrix, (out_w, out_h)


def align_image(img, target_longest=None):
    """
    Detects the PCB boundary in `img` and warps it to a top-down view,
    preserving the board's aspect ratio (longest side = ALIGN_TARGET).

    Args:
        img: OpenCV BGR image (numpy array).

    Returns:
        Aligned BGR image (numpy array), or None if the board boundary
        could not be resolved to a clean quadrilateral.
    """
    return align_image_with_matrix(img, target_longest)[0]


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


def _is_definition_cell(src):
    """
    True if `src` is a "definition-only" notebook cell: it defines at least one
    function/class and its top-level statements are only definitions, module-level
    constants, imports, docstrings and progress prints.

    Demo / batch cells (loops, plots, dataset loading, file writes) and plain
    assignment cells are skipped, so loading Student 1's notebook has no side
    effects. Executing every definition cell - not just the three wired functions -
    is what keeps Student 1's helpers (prepare_board, crop_pcb, find_pcb_regions,
    _pcb_mask, TARGET_SIZE, SAT_THRESHOLD, ...) available to the functions that
    Student 2 calls, however Student 1 renames or extends them.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False

    has_def = False
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            has_def = True
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.Import, ast.ImportFrom, ast.Pass)):
            continue
        elif isinstance(node, ast.Expr):
            value = node.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                continue                          # docstring
            if (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                    and value.func.id == "print"):
                continue                          # harmless progress print
            return False                          # any other top-level call: skip
        else:
            return False                          # loops / conditionals / etc.: skip
    return has_def


def load_student1_functions():
    """
    Dynamically load Student 1's functions directly from their notebook (single
    source of truth = Student 1's notebook). Every "definition-only" cell is
    executed in notebook order, so the three wired functions AND the helpers they
    depend on (prepare_board / crop_pcb / find_pcb_regions / _pcb_mask ...) are all
    available. Demo/batch cells that run loops or plots are skipped and stdout is
    suppressed. Cached after the first call.

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
        if not _is_definition_cell(src):
            continue
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
    if img is None:
        return None                                           # S1 returned undecodable bytes
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


# --------------------------------------------------------------------------- #
# Geometry helpers : carry annotations through the same transforms
# --------------------------------------------------------------------------- #
# The detector must be trained on exactly what Student 2 hands it at run time.
# That means the ground-truth boxes have to travel through the same geometry as
# the pixels: the rotation applied when PCB_DATASET/rotation was generated, and
# Student 2's perspective warp. These helpers do that mapping.

def transform_points(points, matrix):
    """
    Map Nx2 points through a 2x3 affine matrix or a 3x3 homography.

    Args:
        points: sequence of (x, y).
        matrix: 2x3 (affine, e.g. a rotation) or 3x3 (homography).

    Returns:
        Nx2 float32 array of mapped points.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape == (2, 3):
        out = cv2.transform(pts, matrix)
    else:
        out = cv2.perspectiveTransform(pts, matrix)
    return out.reshape(-1, 2)


def transform_boxes(boxes, matrix, out_w, out_h, min_size=2.0):
    """
    Map axis-aligned [x1, y1, x2, y2] boxes through `matrix`.

    All four corners are mapped and the axis-aligned hull is taken, because a
    rotation or a perspective warp turns a rectangle into a quadrilateral. Boxes
    are clipped to the output image and dropped if they shrink below `min_size`
    or fall outside it entirely.

    Returns:
        (mapped_boxes, kept_indices) - `kept_indices` lets the caller drop the
        matching class labels for boxes that were discarded.
    """
    mapped, kept = [], []
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        corners = transform_points(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], matrix)
        nx1, ny1 = corners.min(axis=0)
        nx2, ny2 = corners.max(axis=0)
        nx1 = float(max(0.0, min(nx1, out_w)))
        nx2 = float(max(0.0, min(nx2, out_w)))
        ny1 = float(max(0.0, min(ny1, out_h)))
        ny2 = float(max(0.0, min(ny2, out_h)))
        if (nx2 - nx1) >= min_size and (ny2 - ny1) >= min_size:
            mapped.append([nx1, ny1, nx2, ny2])
            kept.append(i)
    return mapped, kept


def rotation_matrix_bound(width, height, angle):
    """
    Reproduce the rotation used to build PCB_DATASET/rotation (`rotate.py`:
    `rotate_bound_white_bg`), returning the matrix instead of the image so the
    annotations can be rotated with it.

    Args:
        width, height: size of the ORIGINAL (unrotated) image.
        angle: the angle recorded in PCB_DATASET/rotation/<class>_angles.txt.

    Returns:
        (M_2x3, new_width, new_height) - the canvas grows to fit the rotation,
        exactly as rotate.py does, so `new_width/new_height` should match the
        size of the file in the rotation folder.
    """
    c_x, c_y = width // 2, height // 2
    matrix = cv2.getRotationMatrix2D((c_x, c_y), -angle, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_w = int((height * sin) + (width * cos))
    new_h = int((height * cos) + (width * sin))
    matrix[0, 2] += (new_w / 2) - c_x
    matrix[1, 2] += (new_h / 2) - c_y
    return matrix, new_w, new_h


def rotate_bound_white_bg(image, angle):
    """`rotate.py`'s rotation, kept here so the whole chain lives in one module."""
    matrix, new_w, new_h = rotation_matrix_bound(image.shape[1], image.shape[0], angle)
    return cv2.warpAffine(image, matrix, (new_w, new_h), borderValue=(143, 148, 151))


def read_rotation_angles(angles_txt):
    """
    Parse a PCB_DATASET/rotation/<class>_angles.txt file.

    Returns:
        dict {file_stem: angle_in_degrees}
    """
    angles = {}
    with open(angles_txt, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                angles[parts[0]] = float(parts[-1])
    return angles


# --------------------------------------------------------------------------- #
# THE detector input : one function used by training AND by inference
# --------------------------------------------------------------------------- #
def detection_input(raw_bgr, target_longest=None, fallback=True):
    """
    Turn a RAW acquisition image into exactly the array the defect detector
    expects: Student 1's preprocessing followed by Student 2's alignment.

    This is deliberately the ONLY entry point used both when the training set is
    built and when a board is inspected, so the detector can never be shown a
    kind of image it was not trained on. If this function changes, the training
    set must be rebuilt.

    Args:
        raw_bgr: raw BGR image straight from the camera / dataset folder.
        target_longest: override ALIGN_TARGET (used when rebuilding the dataset).
        fallback: if the board outline cannot be found, resize the whole
                  preprocessed frame instead of returning None, so inference
                  degrades gracefully rather than silently detecting nothing.

    Returns:
        (image_for_detector, M_3x3, (out_w, out_h)).
        `M` maps coordinates in `raw_bgr` to coordinates in the returned image.
        Returns (None, None, None) only when `fallback=False` and alignment failed.
    """
    if raw_bgr is None:
        return None, None, None

    target = target_longest or ALIGN_TARGET
    preprocessed = preprocess_image(raw_bgr)
    if preprocessed is None:
        return None, None, None

    aligned, matrix, size = align_image_with_matrix(preprocessed, target)
    if aligned is not None:
        return aligned, matrix, size

    if not fallback:
        return None, None, None

    # Fallback: no clean quadrilateral -> keep the whole frame, scaled to target
    h, w = preprocessed.shape[:2]
    scale = target / max(w, h)
    out_w, out_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(preprocessed, (out_w, out_h), interpolation=cv2.INTER_AREA)
    matrix = np.array([[scale, 0.0, 0.0],
                       [0.0, scale, 0.0],
                       [0.0, 0.0, 1.0]], dtype=np.float32)
    return resized, matrix, (out_w, out_h)


def as_homography(matrix):
    """Promote a 2x3 affine matrix to a 3x3 homography (3x3 passes through)."""
    m = np.asarray(matrix, dtype=np.float32)
    if m.shape == (2, 3):
        m = np.vstack([m, [0.0, 0.0, 1.0]]).astype(np.float32)
    return m


def compose(*matrices):
    """
    Compose transforms in the order they are APPLIED.

    `compose(R, H)` means "first rotate, then warp", i.e. the matrix H @ R.
    Composing first and mapping once is more accurate than mapping a box twice,
    because each mapping of an axis-aligned box has to take a hull and would
    otherwise inflate the box at every step.
    """
    out = np.eye(3, dtype=np.float32)
    for matrix in matrices:
        out = as_homography(matrix) @ out
    return out


def align_from_source(image_path, raw=True, target_longest=None):
    """
    Read an image from disk and return Student 2's aligned board.

    Args:
        image_path: path to the image.
        raw: True  -> the file is a RAW acquisition image, so Student 1's
                      preprocessing runs first (this is the real pipeline).
             False -> the file already came out of Student 1
                      (e.g. Preprocessed_Dataset), so only alignment runs.
        target_longest: override ALIGN_TARGET.

    Returns:
        Aligned BGR array, or None if the board outline was not found.

    Why this exists: the batch step used to call `process_from_folder(cls, filename)`,
    which always looks inside Preprocessed_Dataset. When the notebook was pointed at
    PCB_DATASET/rotation the class folders are named `<Class>_rotation`, that lookup
    missed every file, and the whole batch silently reported "not aligned".
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    if raw:
        return detection_input(img, target_longest=target_longest, fallback=False)[0]
    return align_image(img, target_longest)
