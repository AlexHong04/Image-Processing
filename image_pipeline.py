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
                student1_validate(img)          (valid, message) via Student 1's
                                                validate_pcb_image (new in notebook)
    Student 2 : align_image(img)                array  -> aligned array
    Wired     : process_image_array / process_image_bytes / process_video_bytes
                (Student 1 -> Student 2, all in-memory)
    Folder    : process_from_folder(cls, filename) reads Preprocessed_Dataset
                (Student 1's output folder) and returns the aligned array.

Usage from Student 2 / Student 3 notebooks:
    from image_pipeline import (
        process_image_array, process_image_bytes, process_video_bytes,
        process_from_folder, align_image, preprocess_image,
        student2_from_student1_array, student2_from_student1_bytes,
        student2_from_student1_video, student1_validate,
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
def _largest_contour(binary_mask):
    """Return the largest external contour of a binary mask (or ``None``)."""
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _clipped_board_corners(binary_mask, image_shape):
    """
    Best-effort quadrilateral for a board that is CUT OFF by the frame edge.

    A clipped board has no closed four-corner boundary — its contour runs
    along the image border — so ``approxPolyDP`` can never resolve a clean
    quad. The minimum-area rotated rectangle of the largest contour still
    carries the board's orientation, so using its four corners lets the
    perspective transform straighten the visible part of the board.

    Returns ``(corners_4x2 float32, coverage)`` or ``(None, coverage)``.
    """
    largest = _largest_contour(binary_mask)
    if largest is None:
        return None, 0.0
    area = cv2.contourArea(largest) / (image_shape[0] * image_shape[1])
    (_, _), (rect_w, rect_h), _angle = cv2.minAreaRect(largest)
    if rect_w <= 4 or rect_h <= 4:
        return None, area
    return cv2.boxPoints(cv2.minAreaRect(largest)).astype("float32"), area


def _four_corner_contour(binary_mask, image_shape):
    """
    Detect the largest external contour in `binary_mask` and approximate it to
    exactly 4 corners (the Otsu corner-detection core used by
    :func:`board_corners_threshold`).

    Returns ``(corners_4x2 float32, coverage)`` or ``(None, coverage)`` where
    coverage is the contour area as a fraction of the image, so callers can
    tell a real board from a mask that has merged with the outer frame.
    """
    largest = _largest_contour(binary_mask)
    if largest is None:
        return None, 0.0

    area = cv2.contourArea(largest) / (image_shape[0] * image_shape[1])
    perim = cv2.arcLength(largest, True)

    # Try a few epsilon factors to get exactly 4 corners
    for eps_factor in (0.02, 0.01, 0.03, 0.04):
        approx = cv2.approxPolyDP(largest, eps_factor * perim, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype("float32"), area

    return _clipped_board_corners(binary_mask, image_shape)


def _looks_like_border_quad(corners, height, width, tol_frac=0.03):
    """
    True when `corners` (4 points) describe an axis-aligned rectangle that
    hugs the image border — i.e. the OUTER FRAME / border ring of the crop
    rather than the board. Aligning to such a rectangle is a no-op, which is
    exactly the symptom of a crop that carries the conveyor/desk border.

    Detection is order-independent: the points are checked against the four
    corners of their own bounding box, so a shape like
    (11,11) (11,540) (540,540) (540,11) matches, while a tilted board quad
    such as (28,10) (10,471) (475,501) (501,43) does not.
    """
    if corners is None:
        return False
    pts = np.asarray(corners, dtype=np.float32)
    xmin, xmax = float(pts[:, 0].min()), float(pts[:, 0].max())
    ymin, ymax = float(pts[:, 1].min()), float(pts[:, 1].max())
    tol = max(6, int(round(tol_frac * min(height, width))))
    for tx, ty in ((xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)):
        if np.min(np.abs(pts[:, 0] - tx) + np.abs(pts[:, 1] - ty)) > tol:
            return False
    return True


def _has_outer_ring(img, dark_thr=60, min_frac=0.005):
    """
    True when the image carries a substantial DARK border line: a connected
    dark component that touches the image border and covers at least
    ``min_frac`` of the frame. Black/grey conveyor and desk borders show up
    here; a bare white background or a board that merely reaches the frame
    edge does not, so this never fires on the rotation dataset.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dark = (gray < dark_thr).astype(np.uint8)
    n, labels = cv2.connectedComponents(dark)
    if n <= 1:
        return False
    h, w = gray.shape
    border_labels = (set(labels[0, :]) | set(labels[-1, :])
                     | set(labels[:, 0]) | set(labels[:, -1]))
    border_labels.discard(0)
    if not border_labels:
        return False
    total = h * w
    return any(int(np.sum(labels == lab)) >= min_frac * total
               for lab in border_labels)


def rectify_frame(img, pad=6):
    """
    Rectify EVERY board of a multi-board frame IN PLACE.

    Each board is warped onto the axis-aligned rectangle that currently bounds
    it, so the frame keeps its original layout — the operator still sees the
    conveyor picture, but every board is at its correct angle. This is what the
    video and live pages want, as opposed to cropping the boards out and
    composing them side by side.

    Args:
        img: OpenCV BGR frame (already pre-processed by Module 1, if used).
        pad: padding to add around each detected board before corner detection.

    Returns:
        ``(rectified, num_boards, notes)``. ``rectified`` is ``None`` for
        frames with zero or one board, so callers fall back to the usual
        whole-image alignment; ``notes`` explains any board that could not be
        straightened.
    """
    if img is None:
        return None, 0, ["No image supplied."]

    h, w = img.shape[:2]
    boxes = find_boards(img)
    if not boxes:
        return None, 0, ["No board was detected in the frame."]
    if len(boxes) == 1:
        return None, 1, []

    result = img.copy()
    notes: list[str] = []
    for i, (x, y, bw, bh) in enumerate(boxes):
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
        crop = img[y0:y1, x0:x1]

        corners, _ = board_corners_threshold(crop)
        if corners is None:
            notes.append(f"Board {i + 1} boundary could not be resolved; left as seen.")
            continue

        # Corners are in crop coordinates; bring them into full-frame space and
        # warp the board onto its own axis-aligned bounding rectangle.
        src = order_points(corners + np.array([x0, y0], dtype=np.float32))
        dst = np.float32([
            [x, y],
            [x + bw, y],
            [x + bw, y + bh],
            [x, y + bh],
        ])
        matrix = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(img, matrix, (w, h))

        mask = np.zeros((h, w), np.uint8)
        cv2.rectangle(mask, (x, y), (x + bw, y + bh), 255, -1)
        result[mask > 0] = warped[mask > 0]

    return result, len(boxes), notes


def _saturation_mask(img, sat_threshold=30):
    """
    Colour mask of the board for the fallback path: the PCB is green (high
    saturation) while grey/belt/desk backgrounds and black borders sit near
    zero, so this mask survives whatever brightness does. Same threshold and
    morphology Student 1 uses in ``_pcb_mask`` (SAT_THRESHOLD = 30).
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = (hsv[:, :, 1] > sat_threshold).astype(np.uint8) * 255
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), iterations=3,
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )
    return mask


def find_boards(img, min_area_frac=0.005, max_boards=10):
    """
    Locate every PCB board in a frame — multi-board conveyor frames included.

    The saturation mask separates the green boards from the grey conveyor and
    black border (both sit near zero saturation), so the result is independent
    of brightness. The size and aspect filters are the same ones Student 1's
    ``find_pcb_regions`` applies, so the two agree on where the boards are.

    Args:
        img: OpenCV BGR frame.
        min_area_frac: ignore contours smaller than this fraction of the frame.
        max_boards: stop after this many boxes (left to right).

    Returns:
        A list of ``(x, y, w, h)`` bounding boxes, ordered left to right.
    """
    h, w = img.shape[:2]
    mask = _saturation_mask(img)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw * bh < min_area_frac * w * h:
            continue
        if not (0.4 < bw / float(bh) < 2.5):
            continue
        boxes.append((int(x), int(y), int(bw), int(bh)))

    boxes.sort(key=lambda b: b[0])
    return boxes[:max_boards]


def board_corners_threshold(img):
    """
    Detect the 4 corner points of the PCB board using Otsu thresholding.
    The board is separated from the background by brightness, so thresholding
    is far more reliable than Canny (which fragments the board outline and
    locks onto tiny internal features).

    Student 1 crops the board tightly (only a few pixels of padding), so the
    crop can still carry the OUTER BORDER LINE of the conveyor/desk, and the
    surrounding background can have almost the same brightness as the board.
    Otsu then either merges board + background (the largest contour becomes
    the outer image frame, >95% coverage) or locks onto the border ring
    itself (an axis-aligned rectangle hugging the frame) — aligning to either
    is a no-op. When either signature is detected, the outer border line is
    ignored first: a thin margin is cleared from the mask and the detection is
    retried; if a thick border survives that (it is still an axis-aligned
    border-hugging rectangle), the colour/saturation mask is used instead,
    because grey and black borders have no saturation while the board does.

    Returns (corners_4x2 float32, board coverage) or (None, coverage).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu threshold (invert: board = foreground / white)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological close to bridge small gaps in the board outline
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    h, w = thresh.shape
    corners, area = _four_corner_contour(thresh, img.shape)

    # Either the blob is (nearly) the whole image, or the detected quad is an
    # axis-aligned border-hugging rectangle AND a dark border line is present:
    # in both cases the Otsu result is the outer border, not the board.
    if area > 0.95 or (_has_outer_ring(img) and _looks_like_border_quad(corners, h, w)):
        # 1) Ignore the outer border line: clear a thin margin and retry.
        margin = max(2, int(round(0.02 * min(h, w))))
        cleaned = thresh.copy()
        cleaned[:margin, :] = 0
        cleaned[-margin:, :] = 0
        cleaned[:, :margin] = 0
        cleaned[:, -margin:] = 0
        corners, area = _four_corner_contour(cleaned, img.shape)

        # 2) A thick border survives the margin clear (still a border-hugging
        #    rectangle) — switch to the colour mask, which separates the green
        #    board from the border by saturation instead of brightness.
        if corners is None or _looks_like_border_quad(corners, h, w):
            sat_corners, sat_area = _four_corner_contour(_saturation_mask(img), img.shape)
            if sat_corners is not None:
                corners, area = sat_corners, sat_area

    return corners, area


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
_STUDENT1_FN_NAMES = (
    "preprocess_image", "process_image_from_bytes", "process_full_video_backend",
    "validate_pcb_image",          # Student 1 added PCB validation to the notebook
)
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
    'preprocess_image', 'process_image_from_bytes', 'process_full_video_backend',
    'validate_pcb_image'.
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

    Student 1 now returns a dict: {"success", "message", "was_cropped", "image"}
    (the "image" value is JPEG bytes), so this wrapper unwraps it. The pre-2026
    bytes return is still accepted defensively in case the notebook is rolled back.

    Args:
        image_bytes: raw image bytes (as Student 1's bytes input expects).

    Returns:
        aligned BGR array (for Student 3), or None on failure.
    """
    s1 = load_student1_functions()
    processed = s1["process_image_from_bytes"](image_bytes)   # Student 1 -> dict
    if processed is None:                                     # S1 hit an exception
        return None
    if isinstance(processed, dict):
        if not processed.get("success"):                     # S1 validation/processing failed
            return None
        jpeg_bytes = processed.get("image")
    else:                                                     # defensive: legacy bytes return
        jpeg_bytes = processed
    if not jpeg_bytes:
        return None
    img = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
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


def student1_validate(raw_img):
    """
    Run Student 1's NEW `validate_pcb_image` (added to their notebook) directly,
    so the pipeline can reject non-PCB uploads before doing any work.

    Args:
        raw_img: raw BGR image (numpy array).

    Returns:
        (valid: bool, message: str) exactly as Student 1's function returns it.

    Raises:
        NotImplementedError: if the checked-out notebook has no
            `validate_pcb_image` (e.g. an older Student 1 branch is loaded).
    """
    fn = load_student1_functions().get("validate_pcb_image")
    if fn is None:
        raise NotImplementedError(
            "Student 1's notebook has no validate_pcb_image() - update the notebook."
        )
    return fn(raw_img)


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
