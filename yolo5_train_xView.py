import os
import shutil
import subprocess
from pathlib import Path
from typing import Tuple

from datasets import load_dataset
from PIL import Image

# Local base directory for data and YOLO outputs
BASE_DIR = Path("/Data/cv_Optical_Flow")

# YOLO-formatted dataset directories
YOLO_DATA_DIR = BASE_DIR / "xview_yolo"
TRAIN_IMG_DIR = YOLO_DATA_DIR / "images" / "train"
VAL_IMG_DIR = YOLO_DATA_DIR / "images" / "val"
TRAIN_LABELS_DIR = YOLO_DATA_DIR / "labels" / "train"
VAL_LABELS_DIR = YOLO_DATA_DIR / "labels" / "val"

# YOLOv5 clone directory
YOLOV5_DIR = BASE_DIR / "yolov5"

# Final model weights file
FINAL_MODEL_PATH = BASE_DIR / "xview_yolov5.pt"

# Number of classes in xView
NUM_CLASSES = 60

# Training parameters
BATCH_SIZE = 8
EPOCHS = 30
IMG_SIZE = 640


def create_dir_structure() -> None:
    """
    Creates the directory structure for the YOLO dataset format.
    """
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    YOLO_DATA_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_IMG_DIR.mkdir(parents=True, exist_ok=True)
    VAL_IMG_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_LABELS_DIR.mkdir(parents=True, exist_ok=True)
    VAL_LABELS_DIR.mkdir(parents=True, exist_ok=True)


def xywh_to_xyxy(x: float, y: float, w: float, h: float) -> Tuple[float, float, float, float]:
    """
    Converts a bounding box from [x, y, w, h] (top-left + width/height)
    to [xmin, ymin, xmax, ymax].

    Args:
        x: top-left x-coordinate
        y: top-left y-coordinate
        w: width of the bounding box
        h: height of the bounding box

    Returns:
        (xmin, ymin, xmax, ymax)
    """
    xmin = x
    ymin = y
    xmax = x + w
    ymax = y + h
    return xmin, ymin, xmax, ymax


def convert_bbox_to_yolo(
        xmin: float, ymin: float, xmax: float, ymax: float,
        img_width: int, img_height: int
) -> Tuple[float, float, float, float]:
    """
    Converts absolute [xmin, ymin, xmax, ymax] bounding box coordinates
    to normalized YOLO format [x_center, y_center, box_width, box_height].

    Args:
        xmin: minimum x-coordinate
        ymin: minimum y-coordinate
        xmax: maximum x-coordinate
        ymax: maximum y-coordinate
        img_width: width of the image in pixels
        img_height: height of the image in pixels

    Returns:
        (x_center, y_center, box_width, box_height) in normalized [0..1] coordinates
    """
    b_width = xmax - xmin
    b_height = ymax - ymin
    x_center = xmin + (b_width / 2.0)
    y_center = ymin + (b_height / 2.0)

    x_center /= img_width
    y_center /= img_height
    b_width /= img_width
    b_height /= img_height

    return (x_center, y_center, b_width, b_height)


def write_yolo_label(labels_path: Path, class_id: int, bbox: Tuple[float, float, float, float]) -> None:
    """
    Writes one bounding box to a YOLO label file in the format:

        class_id x_center y_center w h

    Args:
        labels_path: path to the .txt file where labels will be written
        class_id: integer representing the object's class
        bbox: (x_center, y_center, w, h) in normalized coordinates
    """
    with open(labels_path, 'a') as f:
        f.write(
            f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n"
        )


def process_split(
        ds_split,
        split_name: str,
        image_dir: Path,
        label_dir: Path
) -> None:
    """
    Converts a split (train or validation) of the dataset to YOLO format:
      1) Saves images to image_dir
      2) Creates .txt files for bounding boxes in label_dir

    Args:
        ds_split: the split subset of the dataset (e.g. ds["train"])
        split_name: "train" or "validation"
        image_dir: directory path where images will be saved
        label_dir: directory path where YOLO labels (.txt) will be saved
    """
    for i, example in enumerate(ds_split):
        # Retrieve or construct a unique ID for the image
        img_id = f"{split_name}_{i}"
        img_filename = f"{img_id}.jpg"

        # Convert image to PIL if it's not already
        if isinstance(example["image"], Image.Image):
            pil_image = example["image"]
        else:
            pil_image = Image.fromarray(example["image"])

        # Retrieve the image's width and height
        img_width = example["width"]
        img_height = example["height"]

        # Save the image
        img_path = image_dir / img_filename
        pil_image.save(img_path)

        # Read bounding box data from the 'objects' field
        # Each bbox is [x, y, w, h]
        bboxes = example["objects"]["bbox"]
        categories = example["objects"]["category"]

        # Prepare the label file
        label_filepath = label_dir / f"{img_id}.txt"
        if label_filepath.exists():
            label_filepath.unlink()

        # Convert each bounding box to YOLO format and write
        for bbox_xywh, cat_id in zip(bboxes, categories):
            x, y, w, h = bbox_xywh
            xmin, ymin, xmax, ymax = xywh_to_xyxy(x, y, w, h)
            yolo_bbox = convert_bbox_to_yolo(xmin, ymin, xmax, ymax, img_width, img_height)

            # If categories are 1-based, do cat_id -= 1
            # cat_id = cat_id - 1  # Uncomment if needed

            write_yolo_label(label_filepath, cat_id, yolo_bbox)

        if i % 500 == 0:
            print(f"[{split_name}] Processed {i} / {len(ds_split)} examples...")


def main() -> None:
    """
    Main execution flow:
      1) Create the directory structure
      2) Load the xView dataset from Hugging Face
      3) Convert each split to YOLO format
      4) Create data.yaml for YOLOv5
      5) Clone YOLOv5 and install requirements
      6) Train YOLOv5
      7) Save final model weights
    """
    # Step A: Create directory structure
    create_dir_structure()

    # Step B: Load dataset from Hugging Face
    print("Loading xView dataset from Hugging Face (HichTala/xview)...")
    ds = load_dataset("HichTala/xview")
    print("Available splits in the dataset:", list(ds.keys()))

    if "train" not in ds:
        raise ValueError("No 'train' split found in the dataset. Please check the dataset structure.")

    # Check if "validation" split is present; if not, split train
    if "validation" not in ds:
        ds_splits = ds["train"].train_test_split(test_size=0.2, seed=42)
        ds = {
            "train": ds_splits["train"],
            "validation": ds_splits["test"]
        }
    else:
        ds = {
            "train": ds["train"],
            "validation": ds["validation"]
        }

    # Step C: Convert dataset to YOLO Format
    print("Converting 'train' split to YOLO format...")
    process_split(ds["train"], "train", TRAIN_IMG_DIR, TRAIN_LABELS_DIR)

    print("Converting 'validation' split to YOLO format...")
    process_split(ds["validation"], "validation", VAL_IMG_DIR, VAL_LABELS_DIR)

    print("Dataset conversion to YOLO format complete.")

    # Step D: Create data.yaml for YOLOv5
    data_yaml_path = YOLO_DATA_DIR / "xview_data.yaml"
    class_names = [f"class_{i}" for i in range(NUM_CLASSES)]
    with open(data_yaml_path, "w") as f:
        f.write(f"path: {YOLO_DATA_DIR}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("names:\n")
        for name in class_names:
            f.write(f"  - {name}\n")

    # Step E: Clone YOLOv5 and install requirements
    if not YOLOV5_DIR.exists():
        print("Cloning YOLOv5 repository...")
        subprocess.run(["git", "clone", "https://github.com/ultralytics/yolov5", str(YOLOV5_DIR)])
    else:
        print("YOLOv5 repository already exists, skipping clone.")

    print("Installing YOLOv5 requirements...")
    subprocess.run(["pip", "install", "-r", str(YOLOV5_DIR / "requirements.txt")])

    # Step F: Train YOLOv5
    print(f"Starting YOLOv5 training for {EPOCHS} epochs, batch size {BATCH_SIZE}, image size {IMG_SIZE}...")
    subprocess.run([
        "python", str(YOLOV5_DIR / "train.py"),
        "--img", str(IMG_SIZE),
        "--batch", str(BATCH_SIZE),
        "--epochs", str(EPOCHS),
        "--data", str(data_yaml_path),
        "--weights", "yolov5s.pt",
        "--project", str(BASE_DIR),
        "--name", "xview_yolo_exp",
        "--exist-ok"
    ])

    # YOLOv5 typically saves best.pt in the run folder
    best_weights = BASE_DIR / "xview_yolo_exp" / "weights" / "best.pt"
    if best_weights.exists():
        shutil.copy(str(best_weights), str(FINAL_MODEL_PATH))
        print(f"Best weights copied to: {FINAL_MODEL_PATH}")

    print("Training complete.")


if __name__ == "__main__":
    main()
