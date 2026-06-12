import xml.etree.ElementTree as ET
import os
import shutil
import random

# CONFIG
BOXES_DIR = r'dataset'
IMAGES_SRC_DIR = r'dataset/images'

TRAIN_IMG_DIR = r'dataset/train/images'
VAL_IMG_DIR   = r'dataset/val/images'

TRAIN_LBL_DIR = r'dataset/train/labels'
VAL_LBL_DIR   = r'dataset/val/labels'

SPLIT_RATIO = 0.8

# ✅ FIXED CLASS MAP (robust)
CLASS_MAP = {
    "free_parking_space": 0,
    "not_free_parking_space": 1,
    "not free parking space": 1,
    "occupied": 1,
    "free": 0
}

for d in [TRAIN_IMG_DIR, VAL_IMG_DIR, TRAIN_LBL_DIR, VAL_LBL_DIR]:
    os.makedirs(d, exist_ok=True)


def polygon_to_yolo(points, img_w, img_h):
    xs, ys = [], []

    for p in points.split(";"):
        x, y = map(float, p.split(","))
        xs.append(x)
        ys.append(y)

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    cx = (xmin + xmax) / 2 / img_w
    cy = (ymin + ymax) / 2 / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h

    return cx, cy, w, h


tree = ET.parse(os.path.join(BOXES_DIR, "annotations.xml"))
root = tree.getroot()

images = root.findall("image")
random.shuffle(images)

split_idx = int(len(images) * SPLIT_RATIO)
train_files = images[:split_idx]
val_files = images[split_idx:]


def process_split(data, img_out, lbl_out, name):
    skipped = 0
    class_count = {0: 0, 1: 0}

    for img in data:
        img_name = img.attrib["name"].split("/")[-1]
        width = int(img.attrib["width"])
        height = int(img.attrib["height"])

        label_lines = []

        for poly in img.findall("polygon"):
            label = poly.attrib["label"].strip().lower()

            cls = CLASS_MAP.get(label)
            if cls is None:
                continue

            cx, cy, w, h = polygon_to_yolo(poly.attrib["points"], width, height)
            label_lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            class_count[cls] += 1

        if not label_lines:
            skipped += 1
            continue

        shutil.copy(
            os.path.join(IMAGES_SRC_DIR, img_name),
            os.path.join(img_out, img_name)
        )

        txt_name = img_name.replace(".png", ".txt")

        with open(os.path.join(lbl_out, txt_name), "w") as f:
            f.write("\n".join(label_lines))

    print(f"\n{name} DONE")
    print("Class distribution:", class_count)
    print("Skipped:", skipped)


process_split(train_files, TRAIN_IMG_DIR, TRAIN_LBL_DIR, "TRAIN")
process_split(val_files, VAL_IMG_DIR, VAL_LBL_DIR, "VAL")

print("\nDataset conversion completed.")