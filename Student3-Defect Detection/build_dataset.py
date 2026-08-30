#!/usr/bin/env python3
"""
build_dataset.py - headless, RESUMABLE version of 00_data_preparation.ipynb.

Same configuration, same seed, same split algorithm and the same
image_pipeline functions as the notebook, so running either produces the
same dataset. This one can be stopped and restarted: finished images are
skipped, and the per-image records are appended to dataset/_records.jsonl.

    python3 build_dataset.py --seconds 160     # work for ~160s, then stop
    python3 build_dataset.py                   # run to completion

Exit codes: 0 = finished (dataset complete), 2 = more work remains, 1 = error.
"""
import argparse, json, random, sys, time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------- configuration
RANDOM_SEED = 42
SPLIT_RATIO = (0.70, 0.20, 0.10)
USE_ROTATED = True
REFRESH_S1_S2 = True

CLASS_NAMES = ["missing_hole", "mouse_bite", "open_circuit",
               "short", "spur", "spurious_copper"]
CLASS_TO_ID = {n: i for i, n in enumerate(CLASS_NAMES)}
CLASS_FOLDERS = ["Missing_hole", "Mouse_bite", "Open_circuit",
                 "Short", "Spur", "Spurious_copper"]


def find_project_root(start: Path) -> Path:
    for c in [start, *start.parents]:
        if (c / "image_pipeline.py").is_file() and (c / "PCB_DATASET").is_dir():
            return c
    raise FileNotFoundError("Could not locate the Image-Processing project root.")


PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)
sys.path.insert(0, str(PROJECT_ROOT))
import image_pipeline as ip                                        # noqa: E402

ALIGN_TARGET = ip.ALIGN_TARGET
IMAGES_DIR = PROJECT_ROOT / "PCB_DATASET" / "images"
ROTATION_DIR = PROJECT_ROOT / "PCB_DATASET" / "rotation"
ANNOT_DIR = PROJECT_ROOT / "Clean_Dataset" / "Annotations"
WORK_DIR = PROJECT_ROOT / "Student3-Defect Detection"
DATASET_DIR = WORK_DIR / "dataset"
YOLO_DIR = DATASET_DIR / "yolo"
COCO_DIR = DATASET_DIR / "coco"
RESULTS_DIR = WORK_DIR / "results"
RECORDS_FILE = DATASET_DIR / "_records.jsonl"
STATE_FILE = DATASET_DIR / "_build_state.json"


def parse_voc(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    decl_w, decl_h = int(size.find("width").text), int(size.find("height").text)
    labels, boxes = [], []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip().lower()
        if name not in CLASS_TO_ID:
            continue
        bb = obj.find("bndbox")
        boxes.append([float(bb.find(t).text) for t in ("xmin", "ymin", "xmax", "ymax")])
        labels.append(CLASS_TO_ID[name])
    return decl_w, decl_h, labels, boxes


def index_boards():
    boards = []
    for folder in CLASS_FOLDERS:
        angles_file = ROTATION_DIR / f"{folder}_angles.txt"
        angles = ip.read_rotation_angles(angles_file) if angles_file.is_file() else {}
        for img_path in sorted((IMAGES_DIR / folder).glob("*.jpg")):
            stem = img_path.stem
            xml_path = ANNOT_DIR / folder / f"{stem}.xml"
            if not xml_path.is_file():
                continue
            decl_w, decl_h, labels, boxes = parse_voc(xml_path)
            if not labels:
                continue
            rot_path = ROTATION_DIR / f"{folder}_rotation" / f"{stem}.jpg"
            boards.append({"stem": stem, "folder": folder, "cls": CLASS_NAMES[labels[0]],
                           "image": img_path, "decl_size": (decl_w, decl_h),
                           "labels": labels, "boxes": boxes,
                           "rot_image": rot_path if rot_path.is_file() else None,
                           "angle": angles.get(stem)})
    return boards


def make_split(boards):
    rng = random.Random(RANDOM_SEED)
    by_class = defaultdict(list)
    for b in boards:
        by_class[b["cls"]].append(b)
    split_of = {}
    for cls, items in sorted(by_class.items()):
        items = sorted(items, key=lambda x: x["stem"])
        rng.shuffle(items)
        n = len(items)
        n_train = int(round(n * SPLIT_RATIO[0]))
        n_val = int(round(n * SPLIT_RATIO[1]))
        for i, b in enumerate(items):
            split_of[b["stem"]] = ("train" if i < n_train
                                   else ("val" if i < n_train + n_val else "test"))
    return split_of


def purge_dir(directory: Path, quarantine: Path):
    """
    Empty `directory`. Falls back to MOVING the files into `quarantine` when the
    filesystem refuses deletes (the Claude desktop bridge mounts folders without
    delete permission), so a stale previous dataset can never contaminate a fresh
    build even where files cannot be removed.
    """
    moved = 0
    for old in list(directory.glob("*")):
        try:
            old.unlink()
        except (PermissionError, OSError):
            quarantine.mkdir(parents=True, exist_ok=True)
            target = quarantine / old.name
            if target.exists():
                target = quarantine / f"{old.stem}_{int(time.time()*1000)}{old.suffix}"
            old.rename(target)
            moved += 1
    return moved


def to_yolo_line(label, box, w, h):
    x1, y1, x2, y2 = box
    return (f"{label} {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} "
            f"{(x2-x1)/w:.6f} {(y2-y1)/h:.6f}")


def finalise(records, boards, split_of, failures):
    """data.yaml + COCO + dataset_info.json + Preprocessed_Dataset refresh + checks."""
    (YOLO_DIR / "data.yaml").write_text(
        "# PCB defect detection on Student 2's ALIGNED output\n"
        "# generated by build_dataset.py / 00_data_preparation.ipynb\n"
        f"path: {YOLO_DIR.as_posix()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        f"nc: {len(CLASS_NAMES)}\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASS_NAMES)), encoding="utf-8")

    categories = [{"id": i + 1, "name": n, "supercategory": "pcb_defect"}
                  for i, n in enumerate(CLASS_NAMES)]
    for split in ("train", "val", "test"):
        images, annotations, ann_id = [], [], 1
        for img_id, r in enumerate([x for x in records if x["split"] == split], start=1):
            images.append({"id": img_id, "file_name": f"{r['name']}.jpg",
                           "width": r["width"], "height": r["height"]})
            for lab, (x1, y1, x2, y2) in zip(r["labels"], r["boxes"]):
                annotations.append({"id": ann_id, "image_id": img_id, "category_id": lab + 1,
                                    "bbox": [round(x1, 2), round(y1, 2),
                                             round(x2 - x1, 2), round(y2 - y1, 2)],
                                    "area": round((x2 - x1) * (y2 - y1), 2), "iscrowd": 0})
                ann_id += 1
        (COCO_DIR / f"instances_{split}.json").write_text(
            json.dumps({"info": {"description": "PCB defects on aligned boards"},
                        "images": images, "annotations": annotations,
                        "categories": categories}), encoding="utf-8")
        print(f"  instances_{split}.json  {len(images):4d} images, {len(annotations):5d} annotations")

    info = {
        "source": "PCB_DATASET/images + PCB_DATASET/rotation via image_pipeline.detection_input()",
        "pipeline": {
            "student1": "preprocess_image (colour normalise -> Gaussian 5x5 -> CLAHE on L, clip 2.0, 8x8)",
            "student2": "align_image_with_matrix (Otsu board contour -> 4-point perspective warp)",
            "align_target": ALIGN_TARGET,
            "variants": ["original"] + (["rotated"] if USE_ROTATED else []),
        },
        "class_names": CLASS_NAMES, "num_classes": len(CLASS_NAMES),
        "image_size": ALIGN_TARGET, "random_seed": RANDOM_SEED,
        "split_ratio": list(SPLIT_RATIO),
        "counts": {s: sum(1 for r in records if r["split"] == s) for s in ("train", "val", "test")},
        "boxes": {s: sum(len(r["labels"]) for r in records if r["split"] == s)
                  for s in ("train", "val", "test")},
        "boards": {s: sum(1 for b in boards if split_of[b["stem"]] == s)
                   for s in ("train", "val", "test")},
        "failures": failures,
        "paths": {"yolo_root": YOLO_DIR.as_posix(),
                  "yolo_yaml": (YOLO_DIR / "data.yaml").as_posix(),
                  "coco_dir": COCO_DIR.as_posix(),
                  "results_dir": RESULTS_DIR.as_posix()},
        "files": {s: [r["name"] for r in records if r["split"] == s]
                  for s in ("train", "val", "test")},
    }
    (DATASET_DIR / "dataset_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print("  dataset_info.json written")

    if REFRESH_S1_S2:
        clean = PROJECT_ROOT / "Clean_Dataset"
        pre = PROJECT_ROOT / "Preprocessed_Dataset"
        n = 0
        for folder in CLASS_FOLDERS:
            (pre / folder).mkdir(parents=True, exist_ok=True)
            for src in sorted((clean / folder).glob("*.jpg")):
                out = ip.preprocess_image(cv2.imread(str(src)))
                if out is not None:
                    cv2.imwrite(str(pre / folder / src.name), out, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    n += 1
        print(f"  Preprocessed_Dataset refreshed: {n} colour images")

    problems = []
    for split in ("train", "val", "test"):
        imgs = {p.stem for p in (YOLO_DIR / "images" / split).glob("*.jpg")}
        lbls = {p.stem for p in (YOLO_DIR / "labels" / split).glob("*.txt")}
        if imgs != lbls:
            problems.append(f"{split}: {len(imgs ^ lbls)} unpaired files")
    board_split = defaultdict(set)
    for r in records:
        board_split[r["name"].replace("_rot", "")].add(r["split"])
    leaks = [b for b, s in board_split.items() if len(s) > 1]
    if leaks:
        problems.append(f"{len(leaks)} boards leak across splits")

    widths = [x2 - x1 for r in records for (x1, y1, x2, y2) in r["boxes"]]
    print(f"\n  images {len(records)} | boxes {sum(len(r['labels']) for r in records)} "
          f"| median box width {np.median(widths):.0f} px")
    print("  class balance:", dict(Counter(CLASS_NAMES[l] for r in records for l in r["labels"])))
    print("  PROBLEMS:" if problems else "  All checks passed.", problems or "")
    return not problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=0, help="stop after this many seconds")
    ap.add_argument("--fresh", action="store_true", help="discard any previous progress")
    args = ap.parse_args()

    for d in (DATASET_DIR, YOLO_DIR, COCO_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    boards = index_boards()
    split_of = make_split(boards)

    if args.fresh:
        quarantine_root = DATASET_DIR / "_replaced_previous_build"
        moved = 0
        for split in ("train", "val", "test"):
            for sub in ("images", "labels"):
                d = YOLO_DIR / sub / split
                d.mkdir(parents=True, exist_ok=True)
                moved += purge_dir(d, quarantine_root / sub / split)
        for f in (RECORDS_FILE, STATE_FILE):
            if f.exists():
                try:
                    f.unlink()
                except (PermissionError, OSError):
                    quarantine_root.mkdir(parents=True, exist_ok=True)
                    f.rename(quarantine_root / f.name)
        print(f"fresh start: previous output cleared"
              + (f" ({moved} old files moved to {quarantine_root.name}/ because this "
                 f"filesystem does not allow deletes - remove that folder yourself)" if moved else ""))

    for split in ("train", "val", "test"):
        for sub in ("images", "labels"):
            (YOLO_DIR / sub / split).mkdir(parents=True, exist_ok=True)
    if REFRESH_S1_S2:
        for folder in CLASS_FOLDERS:
            (PROJECT_ROOT / "Calibrated_Dataset" / folder).mkdir(parents=True, exist_ok=True)

    records, failures, done_names = [], [], set()
    if RECORDS_FILE.exists():
        for line in RECORDS_FILE.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("failed"):
                failures.append([r["name"], r["failed"]])
            else:
                records.append(r)
            done_names.add(r["name"])
    if STATE_FILE.exists():
        pass

    jobs = []
    for b in boards:
        jobs.append((b, "original", b["stem"]))
        if USE_ROTATED and b["rot_image"] is not None and b["angle"] is not None:
            jobs.append((b, "rotated", f"{b['stem']}_rot"))
    todo = [j for j in jobs if j[2] not in done_names]
    print(f"boards {len(boards)} | jobs {len(jobs)} | already done {len(jobs)-len(todo)} "
          f"| remaining {len(todo)}")

    t0 = time.perf_counter()
    handle = RECORDS_FILE.open("a", encoding="utf-8")
    processed = 0
    for b, variant, name in todo:
        if args.seconds and (time.perf_counter() - t0) > args.seconds:
            break
        split = split_of[b["stem"]]
        if variant == "rotated":
            src_path = b["rot_image"]
            pre_matrix, _, _ = ip.rotation_matrix_bound(*b["decl_size"], b["angle"])
        else:
            src_path = b["image"]
            pre_matrix = np.eye(3, dtype=np.float32)

        img = cv2.imread(str(src_path))
        aligned, H, size = ip.detection_input(img, target_longest=ALIGN_TARGET, fallback=False)
        if aligned is None:
            rec = {"name": name, "failed": "alignment failed"}
            handle.write(json.dumps(rec) + "\n"); failures.append([name, rec["failed"]])
            processed += 1
            continue

        out_w, out_h = size
        matrix = ip.compose(pre_matrix, H)
        bxs, keep = ip.transform_boxes(b["boxes"], matrix, out_w, out_h)
        if not bxs:
            rec = {"name": name, "failed": "all boxes outside the aligned board"}
            handle.write(json.dumps(rec) + "\n"); failures.append([name, rec["failed"]])
            processed += 1
            continue
        labels = [b["labels"][i] for i in keep]

        cv2.imwrite(str(YOLO_DIR / "images" / split / f"{name}.jpg"), aligned,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        (YOLO_DIR / "labels" / split / f"{name}.txt").write_text(
            "\n".join(to_yolo_line(l, bx, out_w, out_h) for l, bx in zip(labels, bxs)),
            encoding="utf-8")
        if REFRESH_S1_S2 and variant == "original":
            cv2.imwrite(str(PROJECT_ROOT / "Calibrated_Dataset" / b["folder"] / f"{b['stem']}.jpg"),
                        aligned, [cv2.IMWRITE_JPEG_QUALITY, 95])

        rec = {"name": name, "split": split, "cls": b["cls"], "variant": variant,
               "width": out_w, "height": out_h, "labels": labels, "boxes": bxs,
               "dropped": len(b["labels"]) - len(bxs)}
        handle.write(json.dumps(rec) + "\n")
        records.append(rec)
        processed += 1
        if processed % 25 == 0:
            handle.flush()
            print(f"  {processed}/{len(todo)} this run "
                  f"({time.perf_counter()-t0:.0f}s, {(time.perf_counter()-t0)/processed:.2f}s each)",
                  flush=True)
    handle.close()

    remaining = len(todo) - processed
    print(f"processed {processed} this run | remaining {remaining} | failures {len(failures)}")
    if remaining > 0:
        return 2

    print("\nAll images built - finalising:")
    ok = finalise(records, boards, split_of, failures)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
