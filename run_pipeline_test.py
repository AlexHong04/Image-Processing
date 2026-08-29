#!/usr/bin/env python3
"""
run_pipeline_test.py
====================
Step-by-step test runner for image_pipeline.py.

You give it an image file; it processes it and prints a LOG after every stage:

    READ -> (optional Student 1 preprocess) -> Otsu threshold ->
    board corners -> corner ordering -> destination size ->
    perspective transform -> aligned result (Student 3 saves).

Usage:
    D:/Lwin/Anaconda_Program_FIles/envs/tf_gpu/python.exe run_pipeline_test.py
    D:/Lwin/Anaconda_Program_FIles/envs/tf_gpu/python.exe run_pipeline_test.py path/to/image.jpg
    D:/Lwin/Anaconda_Program_FIles/envs/tf_gpu/python.exe run_pipeline_test.py path/to/image.jpg --preprocess --save out.jpg
"""

import argparse
import os
import random
import sys

import cv2

# Make sure the project root is importable (works from terminal AND Jupyter).
def _find_root():
    d = os.path.abspath(os.getcwd())
    for _ in range(8):
        if os.path.exists(os.path.join(d, "image_pipeline.py")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return os.path.abspath(os.getcwd())

ROOT = _find_root()
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import image_pipeline as ip

# Where the aligned result is saved by default as the "test image".
DEFAULT_TEST_OUT = os.path.join(ROOT, "student3_test_output", "test_aligned.jpg")

def random_rotation_image():
    """Return a random image path from PCB_DATASET/rotation (any class), or None."""
    rot_dir = os.path.join(ROOT, "PCB_DATASET", "rotation")
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    candidates = []
    if os.path.isdir(rot_dir):
        for cls in sorted(os.listdir(rot_dir)):
            cls_dir = os.path.join(rot_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for f in os.listdir(cls_dir):
                if f.lower().endswith(exts):
                    candidates.append(os.path.join(cls_dir, f))
    return random.choice(candidates) if candidates else None


def log(step, message):
    print(f"[{step:>20}] {message}")


def _mkdir_parent(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def run(image_path, do_preprocess, save_path, save_before=None):
    # 0) READ -----------------------------------------------------------------
    log("READ", f"loading image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        log("ERROR", f"could not read image: {image_path}")
        return 1
    log("READ", f"shape={img.shape[1]}x{img.shape[0]} dtype={img.dtype}")

    # 0b) BEFORE ----------------------------------------------------------------
    log("BEFORE", "original image (before Student 1 & Student 2 processing)")
    if save_before:
        _mkdir_parent(save_before)
        cv2.imwrite(save_before, img)
        log("BEFORE", f"saved original to: {save_before}")

    # 1) STUDENT 1 - PREPROCESS ------------------------------------------------
    if do_preprocess:
        log("S1 PREPROCESS", "colour normalisation -> Gaussian filter -> CLAHE (LAB L)")
        img = ip.preprocess_image(img)
        if img is None:
            log("ERROR", "Student 1 preprocess returned None")
            return 1
        log("S1 PREPROCESS", f"done, shape={img.shape[1]}x{img.shape[0]} (size preserved)")
    else:
        log("S1 PREPROCESS", "skipped (input already preprocessed, e.g. Preprocessed_Dataset)")

    # 2) STUDENT 2 - CORNER DETECTION ------------------------------------------
    log("S2 CORNERS", "Otsu threshold + morphological close + largest contour + approxPolyDP")
    corners, coverage = ip.board_corners_threshold(img)
    if corners is None:
        log("ERROR", f"no clean 4-corner board found (coverage={coverage:.3f})")
        return 1
    log("S2 CORNERS", f"4 corners found, board coverage={coverage:.3f}")
    for i, (x, y) in enumerate(corners):
        log("S2 CORNERS", f"  corner {i}: ({x:.1f}, {y:.1f})")

    # 3) STUDENT 2 - CORNER ORDERING -------------------------------------------
    ordered = ip.order_points(corners)
    for name, (x, y) in zip(["TL", "TR", "BR", "BL"], ordered):
        log("S2 ORDER", f"{name} = ({x:.1f}, {y:.1f})")

    # 4) STUDENT 2 - DESTINATION SIZE (aspect preserved) ------------------------
    dst, out_w, out_h = ip.compute_destination(ordered)
    log("S2 DEST", f"output size = {out_w} x {out_h}  (aspect preserved, longest={ip.ALIGN_TARGET})")

    # 5) STUDENT 2 - PERSPECTIVE TRANSFORM --------------------------------------
    matrix = cv2.getPerspectiveTransform(ordered, dst)
    log("S2 WARP", "getPerspectiveTransform -> warpPerspective")
    aligned = cv2.warpPerspective(img, matrix, (out_w, out_h))
    log("S2 WARP", f"aligned shape={aligned.shape[1]}x{aligned.shape[0]} dtype={aligned.dtype}")

    # 6) RESULT + SAVE (Student 3's step) ---------------------------------------
    log("RESULT", "aligned picture ready for Student 3 to run defect detection")
    if save_path:
        ok = ip.save_aligned(aligned, save_path)
        log("S3 SAVE", f"save_aligned -> {save_path}  ({'OK' if ok else 'FAILED'})")
    else:
        log("S3 SAVE", "skipped (no --save given; Student 3 would save here)")

    print("\nDone.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Step-by-step image pipeline test")
    parser.add_argument("image", nargs="?", default=None,
                        help="path to the input image (default: random image from PCB_DATASET/rotation)")
    parser.add_argument("--preprocess", action="store_true",
                        help="run Student 1 preprocessing on the raw image")
    parser.add_argument("--save", default=None,
                        help="optional path to save the aligned result")
    parser.add_argument("--save-before", default=None,
                        help="optional path to save the ORIGINAL image before processing")

    # Jupyter's `%run` appends the ipykernel connection args  '-f <file>'  to
    # sys.argv, which argparse would reject. Strip them so `%run` works here.
    raw_argv = sys.argv[1:]
    argv = []
    i = 0
    while i < len(raw_argv):
        if raw_argv[i] == "-f":
            i += 2          # skip '-f' and its value (the connection file)
            continue
        argv.append(raw_argv[i])
        i += 1

    args = parser.parse_args(argv)

    image = args.image or random_rotation_image()
    if image is None:
        print("No image found under PCB_DATASET/rotation.")
        return 1
    print(f"Random image chosen: {image}\n")
    # Always save the aligned result as a test image (override with --save).
    save_path = args.save or DEFAULT_TEST_OUT
    return run(image, args.preprocess, save_path, args.save_before)


if __name__ == "__main__":
    sys.exit(main())
