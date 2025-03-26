import subprocess
import os
import cv2 as cv
import numpy as np

def compute_spynet_flow(img1_path: str,
                        img2_path: str,
                        output_dir: str,
                        model: str = "sintel-final",
                        repo_root: str = None):
    """
    Runs SPyNet CLI on two input images, saves raw .flo and color preview,
    and returns (flow, color_bgr).

    Returns:
        flow (np.ndarray): HxWx2 optical flow
        color (np.ndarray): HxWx3 BGR visualization
    """
    if repo_root is None:
        repo_root = os.path.join(os.path.dirname(__file__), "pytorch-spynet")
    run_script = os.path.join(repo_root, "run.py")

    os.makedirs(output_dir, exist_ok=True)
    flo_path = os.path.join(output_dir, "flow.flo")
    png_path = os.path.join(output_dir, "flow_color.png")

    cmd = [
        "python", run_script,
        "--model", model,
        "--one", img1_path,
        "--two", img2_path,
        "--out", flo_path
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"SPyNet failed:\n{proc.stderr}")

    flow = cv.readOpticalFlow(flo_path)
    if flow is None:
        raise RuntimeError(f"Unable to load flow file at {flo_path}")

    # Convert to color
    mag, ang = cv.cartToPolar(flow[...,0], flow[...,1], angleInDegrees=True)
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[...,1] = 255
    hsv[...,0] = (ang/2).astype(np.uint8)
    hsv[...,2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    color = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    cv.imwrite(png_path, color)
    return flow, color


flow, colored = compute_spynet_flow(
    img1_path="extracted_frames/frame_0005.jpg",
    img2_path="extracted_frames/frame_0011.jpg",
    output_dir="output/pytorch_spynet"
)

print("Flow shape:", flow.shape)
