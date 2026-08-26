"""
convert_to_yolo.py
──────────────────
Converts the class and instance PNG maps produced by generate_dataset.py
into Ultralytics YOLO-seg label format, then builds the standard split.

Ultralytics YOLO-seg label format (one .txt per image):
    <class_id>  x1 y1  x2 y2  ...  xn yn
    • class_id : integer  (0 = traversable, 1 = obstacle, 2 = target)
  • coordinates are NORMALISED to [0, 1] (divided by image W / H)
    • each line describes ONE polygon contour of ONE visible instance

What this script does:
  1. Reads class maps from OUTPUT_DIR/masks/ and instance maps from
      OUTPUT_DIR/instances/
  2. Finds contours independently for every visible instance ID
  3. Simplifies each contour with the Douglas–Peucker algorithm
  4. Writes one YOLO label .txt per image
  5. Splits images + labels 80 / 20 into train / val sets
  6. Writes dataset.yaml for Ultralytics

Usage
-----
    pip install opencv-python numpy pyyaml          # system / venv python
    python convert_to_yolo.py

Edit the CONFIG block below to match your paths.
"""

import os
import cv2
import random
import shutil
import numpy as np
import yaml
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════
#  CONFIG  ← must match what you set in generate_dataset.py
# ═══════════════════════════════════════════════════════════════════

# Top-level directory that contains images/ and masks/
OUTPUT_DIR = "/home/dho/work/ileri_otonom/seg_dataset_gen/output"

# YOLO dataset root (train/ val/ sub-dirs will be created here)
YOLO_DIR = "/home/dho/work/ileri_otonom/seg_dataset_gen/yolo_dataset"

# Class definitions — must be consistent with what you use in training
# Index in this list == class id in the label files
CLASSES = ["traversable", "obstacle", "target"]

# Train / validation split ratio
TRAIN_RATIO = 0.80

# Contour simplification: higher ε → fewer polygon points (faster training).
# Expressed as a fraction of the contour arc length.
# 0.005 is a good starting point; increase to 0.01–0.02 for sparser polygons.
CONTOUR_EPSILON_FRACTION = 0.005

# Minimum contour area in pixels to keep (filters out tiny noise blobs)
MIN_CONTOUR_AREA_PX = 50

# Random seed for reproducible train/val split
RANDOM_SEED = 42

# ═══════════════════════════════════════════════════════════════════


def find_and_simplify_contours(mask_gray: np.ndarray,
                               target_value: int,
                               w: int, h: int,
                               epsilon_frac: float,
                               min_area: int) -> list:
    """
    Find external contours of regions with *target_value* in *mask_gray*,
    simplify them, and return a list of normalised polygon arrays.

    Each element of the returned list is a flat list:
        [x1/w, y1/h, x2/w, y2/h, ...]   (all in [0, 1])
    """
    # Threshold to binary
    binary = np.zeros(mask_gray.shape, dtype=np.uint8)
    binary[mask_gray == target_value] = 255

    # Optional: small morphological closing to fill pixel-level gaps
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary  = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Find external contours
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    polys = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue    # ignore tiny noise blobs

        # Douglas–Peucker simplification
        arc_len = cv2.arcLength(cnt, closed=True)
        eps     = epsilon_frac * arc_len
        approx  = cv2.approxPolyDP(cnt, eps, closed=True)

        # Need at least 3 points for a valid polygon
        if len(approx) < 3:
            continue

        # Flatten and normalise
        pts = approx.reshape(-1, 2)
        norm = []
        for (px, py) in pts:
            norm.append(round(float(px) / w, 6))
            norm.append(round(float(py) / h, 6))

        polys.append(norm)

    return polys


def instance_maps_to_yolo_lines(class_path: str,
                       instance_path: str,
                       epsilon_frac: float,
                       min_area: int) -> list:
    """
    Convert per-pixel class and instance maps into one YOLO polygon per
    visible instance contour. Instance IDs are zero for background.
    """
    class_map = cv2.imread(class_path, cv2.IMREAD_GRAYSCALE)
    instance_map = cv2.imread(instance_path, cv2.IMREAD_UNCHANGED)
    if class_map is None:
        raise FileNotFoundError(f"Could not read class map: {class_path}")
    if instance_map is None:
        raise FileNotFoundError(f"Could not read instance map: {instance_path}")
    if class_map.shape != instance_map.shape:
        raise ValueError(f"Map shape mismatch for {Path(class_path).stem}")

    h, w = class_map.shape
    lines = []

    for instance_id in np.unique(instance_map):
        if instance_id == 0:
            continue
        instance_pixels = instance_map == instance_id
        class_values = class_map[instance_pixels]
        if class_values.size == 0:
            continue
        class_id = int(np.bincount(class_values).argmax())
        if class_id >= len(CLASSES):
            continue
        polygons = find_and_simplify_contours(
            instance_map, int(instance_id), w, h, epsilon_frac, min_area
        )
        for poly in polygons:
            lines.append(f"{class_id} {' '.join(map(str, poly))}")

    return lines


def build_yolo_dataset(output_dir: str,
                       yolo_dir: str,
                       classes: list,
                       train_ratio: float,
                       epsilon_frac: float,
                       min_area: int,
                       seed: int) -> None:

    img_src  = os.path.join(output_dir, "images")
    mask_src = os.path.join(output_dir, "masks")
    instance_src = os.path.join(output_dir, "instances")

    if not os.path.isdir(img_src):
        raise FileNotFoundError(f"images/ directory not found: {img_src}")
    if not os.path.isdir(mask_src):
        raise FileNotFoundError(f"masks/ directory not found: {mask_src}")
    if not os.path.isdir(instance_src):
        raise FileNotFoundError(f"instances/ directory not found: {instance_src}")

    # Collect all sample stems (e.g. "0000", "0001", …)
    stems = sorted(
        Path(p).stem
        for p in os.listdir(img_src)
        if p.lower().endswith(".png")
    )
    if not stems:
        raise RuntimeError(f"No PNG files found in {img_src}")

    print(f"[INFO] Found {len(stems)} image–mask pairs.")

    # Train / val split
    random.seed(seed)
    shuffled   = stems.copy()
    random.shuffle(shuffled)
    n_train    = int(len(shuffled) * train_ratio)
    train_stems = shuffled[:n_train]
    val_stems   = shuffled[n_train:]
    print(f"[INFO] Split → train: {len(train_stems)}, val: {len(val_stems)}")

    # Create directory structure
    for split in ("train", "val"):
        os.makedirs(os.path.join(yolo_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(yolo_dir, "labels", split), exist_ok=True)

    # Process each split
    total_labels = 0
    empty_labels = 0

    for split, stem_list in [("train", train_stems), ("val", val_stems)]:
        for stem in stem_list:
            src_img  = os.path.join(img_src,  f"{stem}.png")
            src_mask = os.path.join(mask_src, f"{stem}.png")
            src_instance = os.path.join(instance_src, f"{stem}.png")

            # Copy image
            dst_img = os.path.join(yolo_dir, "images", split, f"{stem}.png")
            shutil.copy2(src_img, dst_img)

            # Generate YOLO label
            lines = instance_maps_to_yolo_lines(
                src_mask, src_instance, epsilon_frac, min_area
            )

            dst_lbl = os.path.join(yolo_dir, "labels", split, f"{stem}.txt")
            with open(dst_lbl, "w") as f:
                f.write("\n".join(lines))

            total_labels += len(lines)
            if not lines:
                empty_labels += 1
                print(f"  [WARN] {stem}.png → no contours found "
                      "(mask may be all-ground or all-obstacle)")

    print(f"\n[INFO] Label lines written: {total_labels}")
    if empty_labels:
        print(f"[WARN] Empty label files:   {empty_labels} "
              "(images with no obstacles — valid for background images)")

    # Write dataset.yaml
    yaml_path = os.path.join(yolo_dir, "dataset.yaml")
    dataset_cfg = {
        "path"  : yolo_dir,
        "train" : "images/train",
        "val"   : "images/val",
        "nc"    : len(classes),
        "names" : classes,
    }
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_cfg, f, default_flow_style=False, sort_keys=False)

    print(f"\n[DONE] YOLO dataset written to: {yolo_dir}")
    print(f"       dataset.yaml: {yaml_path}")
    print(
        "\nTo train Ultralytics YOLO11-seg:\n"
        f"    yolo segment train data={yaml_path} "
        "model=yolo11n-seg.pt epochs=100 imgsz=640"
    )


def verify_label(label_path: str, image_path: str) -> None:
    """
    Quick sanity-check: draw segmentation polygons on the image and
    save a side-by-side debug PNG next to the label file.
    Only called when DEBUG=True below.
    """
    img  = cv2.imread(image_path)
    h, w = img.shape[:2]
    colours = {0: (0, 200, 0), 1: (0, 0, 255), 2: (255, 0, 0)}

    with open(label_path) as f:
        for line in f:
            parts   = line.strip().split()
            if not parts:
                continue
            cls_id  = int(parts[0])
            coords  = list(map(float, parts[1:]))
            pts     = np.array(
                [[int(coords[k]*w), int(coords[k+1]*h)]
                 for k in range(0, len(coords), 2)],
                dtype=np.int32
            )
            colour  = colours.get(cls_id, (255, 255, 0))
            cv2.polylines(img, [pts], isClosed=True, color=colour, thickness=2)

    debug_path = label_path.replace(".txt", "_debug.jpg")
    cv2.imwrite(debug_path, img)
    print(f"  debug image → {debug_path}")


# ═══════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════

# Set DEBUG = True to generate a handful of overlay debug images
DEBUG            = True
DEBUG_MAX_IMAGES = 5

if __name__ == "__main__":
    build_yolo_dataset(
        output_dir          = OUTPUT_DIR,
        yolo_dir            = YOLO_DIR,
        classes             = CLASSES,
        train_ratio         = TRAIN_RATIO,
        epsilon_frac        = CONTOUR_EPSILON_FRACTION,
        min_area            = MIN_CONTOUR_AREA_PX,
        seed                = RANDOM_SEED,
    )

    if DEBUG:
        print("\n[DEBUG] Generating polygon overlay images …")
        lbl_dir = os.path.join(YOLO_DIR, "labels", "train")
        img_dir = os.path.join(YOLO_DIR, "images", "train")
        count   = 0
        for lbl_file in sorted(os.listdir(lbl_dir)):
            if not lbl_file.endswith(".txt"):
                continue
            stem   = Path(lbl_file).stem
            verify_label(
                os.path.join(lbl_dir, lbl_file),
                os.path.join(img_dir, f"{stem}.png"),
            )
            count += 1
            if count >= DEBUG_MAX_IMAGES:
                break
