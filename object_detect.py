import subprocess
from pathlib import Path

def main() -> None:
    """
    Runs YOLOv5 detection on an MP4 video using your trained xView model
    and a data.yaml file containing real xView class names.
    """
    # Paths to your YOLOv5 repo, model weights, data.yaml, and input video
    YOLOV5_DIR = Path("/Data/cv_Optical_Flow/yolov5")         # Cloned YOLOv5 repository
    MODEL_PATH = Path("/Data/cv_Optical_Flow/xview_yolov5.pt") # Trained weights file
    DATA_PATH = Path("/Data/cv_Optical_Flow/xview_data.yaml")  # YAML file with real class names
    VIDEO_PATH = Path("drone_footage/cut1.mp4")                # Path to the MP4 video

    # Define where to save the inference results
    PROJECT = Path("output/object_detection")  # YOLOv5 creates this folder if nonexistent
    NAME = "xview_inference"                   # Subfolder within PROJECT

    # Confidence threshold, image size, etc., can be adjusted
    cmd = [
        "python", str(YOLOV5_DIR / "detect.py"),
        "--weights", str(MODEL_PATH),
        "--source", str(VIDEO_PATH),       # video file, image folder, etc.
        "--conf-thres", "0.25",            # confidence threshold
        "--img-size", "640",               # inference image size
        "--data", str(DATA_PATH),          # <-- use your xview_data.yaml here
        "--project", str(PROJECT),
        "--name", NAME,
        "--exist-ok"                       # overwrite output folder if it exists
    ]

    print("Running YOLOv5 inference on video...")
    subprocess.run(cmd, check=True)
    print("Inference complete. Results are saved in:")
    print(PROJECT / NAME)

if __name__ == "__main__":
    main()
