#!/usr/bin/env python3
"""
PCB Defect Detection - Benchmark Dataset Preparation
=====================================================
Student 3 module: Defect Detection (YOLOv10 vs Faster R-CNN vs RT-DETR)

Runs on the local machine (needs only python3 + opencv + numpy).

Produces:
  DefectDetection_Benchmark/
    dataset_stats.json          full dataset statistics (for the report)
    splits.json                 stratified train/val/test image manifest
    labels/<stem>.txt           YOLO-format labels (normalised -> valid for BOTH variants)
    coco/raw_{split}.json       COCO annotations at native resolution
    coco/pre_{split}.json       COCO annotations at 512x512
    data_raw.yaml               Ultralytics data config (raw variant)
    data_pre.yaml               Ultralytics data config (preprocessed variant)
    build_dataset.py            materialises the YOLO folder tree (run in Colab)
  Preprocessed_Dataset/<class>/*.jpg    Student-1 pipeline output (norm -> LAB -> CLAHE)

Why one shared label set works for both variants
------------------------------------------------
YOLO labels are normalised to [0,1] by image width/height. The 512x512 resize is a
pure (non-aspect-preserving) affine scale, so x/W and y/H are unchanged. Only the
COCO files, which use absolute pixels, need two versions.
"""

import json
import os
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent
RAW_IMG_DIR = ROOT / "PCB_DATASET" / "images"
ANN_DIR = ROOT / "PCB_DATASET" / "Annotations"
CLEAN_DIR = ROOT / "Clean_Dataset"
PRE_DIR = ROOT / "Preprocessed_Dataset"
OUT = ROOT / "DefectDetection_Benchmark"

CLASSES = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper",
]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}

PRE_SIZE = 512          # Student-1 clean/preprocessed resolution
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)
SPLIT = (0.70, 0.15, 0.15)
SEED = 42


# --------------------------------------------------------------------------- #
# 1. Parse the Pascal-VOC annotations
# --------------------------------------------------------------------------- #
def parse_annotations():
    """Return a list of per-image records parsed from the VOC XML files."""
    records = []
    problems = []

    for cls_dir in sorted(ANN_DIR.iterdir()):
        if not cls_dir.is_dir():
            continue
        for xml_path in sorted(cls_dir.glob("*.xml")):
            try:
                root = ET.parse(xml_path).getroot()
            except ET.ParseError as exc:
                problems.append(f"unparseable XML: {xml_path.name} ({exc})")
                continue

            size = root.find("size")
            width = int(size.findtext("width"))
            height = int(size.findtext("height"))
            filename = root.findtext("filename")
            stem = xml_path.stem

            img_path = RAW_IMG_DIR / cls_dir.name / f"{stem}.jpg"
            if not img_path.exists():
                problems.append(f"missing raw image for {stem}")
                continue

            boxes = []
            for obj in root.findall("object"):
                name = (obj.findtext("name") or "").strip().lower()
                if name not in CLASS_TO_ID:
                    problems.append(f"unknown class '{name}' in {xml_path.name}")
                    continue
                bb = obj.find("bndbox")
                xmin = float(bb.findtext("xmin"))
                ymin = float(bb.findtext("ymin"))
                xmax = float(bb.findtext("xmax"))
                ymax = float(bb.findtext("ymax"))

                # clip to image bounds and drop degenerate boxes
                xmin, xmax = max(0.0, min(xmin, xmax)), min(float(width), max(xmin, xmax))
                ymin, ymax = max(0.0, min(ymin, ymax)), min(float(height), max(ymin, ymax))
                if xmax - xmin < 1 or ymax - ymin < 1:
                    problems.append(f"degenerate box in {xml_path.name}")
                    continue

                boxes.append(
                    {
                        "cls": CLASS_TO_ID[name],
                        "cls_name": name,
                        "xmin": xmin,
                        "ymin": ymin,
                        "xmax": xmax,
                        "ymax": ymax,
                    }
                )

            if not boxes:
                problems.append(f"no valid objects in {xml_path.name}")
                continue

            records.append(
                {
                    "stem": stem,
                    "filename": filename,
                    "folder": cls_dir.name,
                    "width": width,
                    "height": height,
                    "boxes": boxes,
                    "image_class": Counter(b["cls_name"] for b in boxes).most_common(1)[0][0],
                }
            )

    return records, problems


# --------------------------------------------------------------------------- #
# 2. Dataset statistics
# --------------------------------------------------------------------------- #
def _pct(arr, q):
    return float(np.percentile(arr, q)) if len(arr) else None


def compute_stats(records, problems):
    per_class_imgs = Counter(r["image_class"] for r in records)
    per_class_objs = Counter()
    boxes_per_img = []
    widths, heights, rel_areas, aspects = [], [], [], []
    per_class_relarea = defaultdict(list)
    resolutions = Counter()
    mixed_class_images = 0

    for r in records:
        resolutions[f'{r["width"]}x{r["height"]}'] += 1
        boxes_per_img.append(len(r["boxes"]))
        if len({b["cls_name"] for b in r["boxes"]}) > 1:
            mixed_class_images += 1
        for b in r["boxes"]:
            per_class_objs[b["cls_name"]] += 1
            bw = b["xmax"] - b["xmin"]
            bh = b["ymax"] - b["ymin"]
            widths.append(bw)
            heights.append(bh)
            ra = (bw * bh) / (r["width"] * r["height"])
            rel_areas.append(ra)
            per_class_relarea[b["cls_name"]].append(ra)
            aspects.append(bw / bh)

    widths_a = np.array(widths)
    heights_a = np.array(heights)
    rel_a = np.array(rel_areas)

    # COCO small/medium/large convention, applied after letterboxing to 640
    # and after the non-aspect-preserving squash to 512x512.
    side_640, side_512 = [], []
    for r in records:
        s640 = 640.0 / max(r["width"], r["height"])          # letterbox, aspect preserved
        sx512 = PRE_SIZE / r["width"]                          # squash
        sy512 = PRE_SIZE / r["height"]
        for b in r["boxes"]:
            bw = b["xmax"] - b["xmin"]
            bh = b["ymax"] - b["ymin"]
            side_640.append(np.sqrt(bw * s640 * bh * s640))
            side_512.append(np.sqrt(bw * sx512 * bh * sy512))
    side_640 = np.array(side_640)
    side_512 = np.array(side_512)

    def coco_bucket(areas_sqrt):
        a = areas_sqrt ** 2
        return {
            "small_<32px2": int((a < 32 ** 2).sum()),
            "medium_32-96px2": int(((a >= 32 ** 2) & (a < 96 ** 2)).sum()),
            "large_>96px2": int((a >= 96 ** 2).sum()),
        }

    return {
        "total_images": len(records),
        "total_objects": int(sum(boxes_per_img)),
        "classes": CLASSES,
        "num_classes": len(CLASSES),
        "images_per_class": {c: per_class_imgs.get(c, 0) for c in CLASSES},
        "objects_per_class": {c: per_class_objs.get(c, 0) for c in CLASSES},
        "class_imbalance_ratio": round(
            max(per_class_objs.values()) / max(1, min(per_class_objs.values())), 3
        ),
        "resolutions": dict(resolutions),
        "mixed_class_images": mixed_class_images,
        "objects_per_image": {
            "mean": round(float(np.mean(boxes_per_img)), 3),
            "min": int(np.min(boxes_per_img)),
            "max": int(np.max(boxes_per_img)),
            "median": float(np.median(boxes_per_img)),
        },
        "box_size_native_px": {
            "width_mean": round(float(widths_a.mean()), 2),
            "height_mean": round(float(heights_a.mean()), 2),
            "width_p5": round(_pct(widths_a, 5), 2),
            "width_p95": round(_pct(widths_a, 95), 2),
            "height_p5": round(_pct(heights_a, 5), 2),
            "height_p95": round(_pct(heights_a, 95), 2),
            "aspect_mean": round(float(np.mean(aspects)), 3),
        },
        "relative_box_area": {
            "mean_pct_of_image": round(float(rel_a.mean()) * 100, 5),
            "median_pct_of_image": round(float(np.median(rel_a)) * 100, 5),
            "max_pct_of_image": round(float(rel_a.max()) * 100, 5),
        },
        "relative_area_pct_per_class": {
            c: round(float(np.mean(per_class_relarea[c])) * 100, 5) for c in CLASSES
        },
        "effective_box_side_px": {
            "letterbox_640_mean": round(float(side_640.mean()), 2),
            "letterbox_640_p5": round(_pct(side_640, 5), 2),
            "squash_512_mean": round(float(side_512.mean()), 2),
            "squash_512_p5": round(_pct(side_512, 5), 2),
        },
        "coco_size_buckets_at_640_letterbox": coco_bucket(side_640),
        "coco_size_buckets_at_512_squash": coco_bucket(side_512),
        "data_quality_issues": problems[:50],
        "data_quality_issue_count": len(problems),
    }


# --------------------------------------------------------------------------- #
# 3. Stratified split
# --------------------------------------------------------------------------- #
def make_splits(records):
    rng = random.Random(SEED)
    by_class = defaultdict(list)
    for r in records:
        by_class[r["image_class"]].append(r["stem"])

    splits = {"train": [], "val": [], "test": []}
    for cls in CLASSES:
        stems = sorted(by_class[cls])
        rng.shuffle(stems)
        n = len(stems)
        n_tr = int(round(n * SPLIT[0]))
        n_va = int(round(n * SPLIT[1]))
        splits["train"] += stems[:n_tr]
        splits["val"] += stems[n_tr:n_tr + n_va]
        splits["test"] += stems[n_tr + n_va:]

    for k in splits:
        splits[k] = sorted(splits[k])

    assert not (set(splits["train"]) & set(splits["val"]))
    assert not (set(splits["train"]) & set(splits["test"]))
    assert not (set(splits["val"]) & set(splits["test"]))
    assert sum(len(v) for v in splits.values()) == len(records)
    return splits


# --------------------------------------------------------------------------- #
# 4. YOLO labels (normalised -> shared by both variants)
# --------------------------------------------------------------------------- #
def write_yolo_labels(records, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in records:
        lines = []
        for b in r["boxes"]:
            cx = ((b["xmin"] + b["xmax"]) / 2) / r["width"]
            cy = ((b["ymin"] + b["ymax"]) / 2) / r["height"]
            bw = (b["xmax"] - b["xmin"]) / r["width"]
            bh = (b["ymax"] - b["ymin"]) / r["height"]
            cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
            bw, bh = min(bw, 1.0), min(bh, 1.0)
            lines.append(f"{b['cls']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        (out_dir / f"{r['stem']}.txt").write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# 5. COCO JSON (absolute pixels -> one per variant)
# --------------------------------------------------------------------------- #
def write_coco(records, splits, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    by_stem = {r["stem"]: r for r in records}

    for variant in ("raw", "pre"):
        for split_name, stems in splits.items():
            images, annotations = [], []
            ann_id = 1
            for img_id, stem in enumerate(stems, start=1):
                r = by_stem[stem]
                if variant == "raw":
                    W, H = r["width"], r["height"]
                    sx = sy = 1.0
                    rel = f'{r["folder"]}/{stem}.jpg'
                else:
                    W = H = PRE_SIZE
                    sx = PRE_SIZE / r["width"]
                    sy = PRE_SIZE / r["height"]
                    rel = f'{r["folder"]}/{stem}.jpg'

                images.append(
                    {"id": img_id, "file_name": rel, "width": W, "height": H}
                )
                for b in r["boxes"]:
                    x = b["xmin"] * sx
                    y = b["ymin"] * sy
                    w = (b["xmax"] - b["xmin"]) * sx
                    h = (b["ymax"] - b["ymin"]) * sy
                    annotations.append(
                        {
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": b["cls"] + 1,   # COCO ids are 1-based
                            "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                            "area": round(w * h, 2),
                            "iscrowd": 0,
                        }
                    )
                    ann_id += 1

            coco = {
                "info": {"description": f"HRIPCB defect detection - {variant} - {split_name}"},
                "images": images,
                "annotations": annotations,
                "categories": [
                    {"id": i + 1, "name": c, "supercategory": "pcb_defect"}
                    for i, c in enumerate(CLASSES)
                ],
            }
            (out_dir / f"{variant}_{split_name}.json").write_text(json.dumps(coco))


# --------------------------------------------------------------------------- #
# 6. Student-1 preprocessing pipeline -> Preprocessed_Dataset
# --------------------------------------------------------------------------- #
def build_preprocessed(records):
    """Min-max normalisation -> BGR2LAB -> CLAHE on L -> LAB2BGR, at 512x512."""
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
    made, skipped = 0, 0

    for r in records:
        dst_dir = PRE_DIR / r["folder"]
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f'{r["stem"]}.jpg'
        if dst.exists():
            skipped += 1
            continue

        src = CLEAN_DIR / r["folder"] / f'{r["stem"]}.jpg'
        if src.exists():
            img = cv2.imread(str(src))
        else:  # fall back to resizing the raw image ourselves
            img = cv2.imread(str(RAW_IMG_DIR / r["folder"] / f'{r["stem"]}.jpg'))
            if img is not None:
                img = cv2.resize(img, (PRE_SIZE, PRE_SIZE), interpolation=cv2.INTER_CUBIC)
        if img is None:
            continue

        if img.shape[0] != PRE_SIZE or img.shape[1] != PRE_SIZE:
            img = cv2.resize(img, (PRE_SIZE, PRE_SIZE), interpolation=cv2.INTER_CUBIC)

        norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        lab = cv2.cvtColor(norm, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        lab = cv2.merge((clahe.apply(l), a, b))
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        cv2.imwrite(str(dst), out, [cv2.IMWRITE_JPEG_QUALITY, 95])
        made += 1

    return made, skipped


# --------------------------------------------------------------------------- #
# 7. Ultralytics data yaml + Colab dataset builder
# --------------------------------------------------------------------------- #
DATA_YAML = """# Ultralytics data config - {variant} variant
# Generated by prepare_benchmark_data.py
path: {root}
train: images/train
val: images/val
test: images/test

nc: 6
names:
{names}
"""

BUILD_SCRIPT = '''#!/usr/bin/env python3
"""
Materialise the YOLO folder tree for one dataset variant.

    python build_dataset.py --variant raw --src /path/to/PCB_DATASET/images --dst ./ds_raw
    python build_dataset.py --variant pre --src /path/to/Preprocessed_Dataset --dst ./ds_pre

Creates:
    <dst>/images/{train,val,test}/*.jpg
    <dst>/labels/{train,val,test}/*.txt
    <dst>/data.yaml
"""
import argparse, json, os, shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLASSES = ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]

ap = argparse.ArgumentParser()
ap.add_argument("--variant", choices=["raw", "pre"], required=True)
ap.add_argument("--src", required=True, help="root folder holding <class>/<stem>.jpg")
ap.add_argument("--dst", required=True)
ap.add_argument("--link", action="store_true", help="symlink instead of copy (saves disk)")
args = ap.parse_args()

splits = json.loads((HERE / "splits.json").read_text())
folders = splits["folder_of_stem"]
src, dst = Path(args.src), Path(args.dst)

for split in ("train", "val", "test"):
    (dst / "images" / split).mkdir(parents=True, exist_ok=True)
    (dst / "labels" / split).mkdir(parents=True, exist_ok=True)
    for stem in splits[split]:
        s = src / folders[stem] / f"{stem}.jpg"
        d = dst / "images" / split / f"{stem}.jpg"
        if not d.exists():
            if args.link:
                os.symlink(s.resolve(), d)
            else:
                shutil.copy2(s, d)
        shutil.copy2(HERE / "labels" / f"{stem}.txt", dst / "labels" / split / f"{stem}.txt")

(dst / "data.yaml").write_text(
    f"path: {dst.resolve()}\\ntrain: images/train\\nval: images/val\\ntest: images/test\\n"
    f"nc: {len(CLASSES)}\\nnames:\\n" + "".join(f"  {i}: {c}\\n" for i, c in enumerate(CLASSES))
)
print(f"[ok] {args.variant}: "
      + ", ".join(f"{s}={len(splits[s])}" for s in ("train", "val", "test"))
      + f" -> {dst}")
'''


# --------------------------------------------------------------------------- #
def main():
    print("=" * 70)
    print("PCB Defect Detection - Benchmark Dataset Preparation")
    print("=" * 70)

    for p in (RAW_IMG_DIR, ANN_DIR):
        if not p.exists():
            sys.exit(f"[fatal] missing required folder: {p}")

    print("\n[1/6] Parsing Pascal-VOC annotations ...")
    records, problems = parse_annotations()
    print(f"      parsed {len(records)} images, {sum(len(r['boxes']) for r in records)} objects")
    if problems:
        print(f"      {len(problems)} data-quality issues (first 5): {problems[:5]}")

    OUT.mkdir(parents=True, exist_ok=True)

    print("\n[2/6] Computing dataset statistics ...")
    stats = compute_stats(records, problems)
    (OUT / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"      objects/class: {stats['objects_per_class']}")
    print(f"      mean defect area: {stats['relative_box_area']['mean_pct_of_image']}% of image")
    print(f"      small objects @640 letterbox: {stats['coco_size_buckets_at_640_letterbox']}")
    print(f"      small objects @512 squash  : {stats['coco_size_buckets_at_512_squash']}")

    print("\n[3/6] Building stratified 70/15/15 split ...")
    splits = make_splits(records)
    splits["folder_of_stem"] = {r["stem"]: r["folder"] for r in records}
    splits["class_of_stem"] = {r["stem"]: r["image_class"] for r in records}
    splits["seed"] = SEED
    (OUT / "splits.json").write_text(json.dumps(splits, indent=2))
    print(f"      train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")

    print("\n[4/6] Writing YOLO labels + COCO annotations ...")
    write_yolo_labels(records, OUT / "labels")
    write_coco(records, {k: splits[k] for k in ("train", "val", "test")}, OUT / "coco")
    print(f"      {len(records)} label files, 6 COCO json files")

    print("\n[5/6] Building Preprocessed_Dataset (norm -> LAB -> CLAHE) ...")
    made, skipped = build_preprocessed(records)
    print(f"      created {made}, already present {skipped} -> {PRE_DIR}")

    print("\n[6/6] Writing configs and Colab builder ...")
    names = "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES))
    (OUT / "data_raw.yaml").write_text(
        DATA_YAML.format(variant="raw", root="./ds_raw", names=names)
    )
    (OUT / "data_pre.yaml").write_text(
        DATA_YAML.format(variant="pre", root="./ds_pre", names=names)
    )
    (OUT / "build_dataset.py").write_text(BUILD_SCRIPT)

    # small zip of everything except the images, for uploading to Colab
    shutil.make_archive(str(ROOT / "DefectDetection_Benchmark"), "zip", root_dir=OUT)
    print(f"      packaged -> {ROOT / 'DefectDetection_Benchmark.zip'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
