import os
import numpy as np
import cv2 as cv

import cv2
import numpy as np

import cv2
import numpy as np


def compute_dense_optical_flow(input_video: str, output_video_path: str, resize_dim: int = 600, fps: float = 10.0) -> None:
    """
    Reads `input_video`, computes dense optical flow on each frame,
    overlays the flow visualization on the original frames, and writes
    the result to `output_video_path` without displaying it.

    Parameters
    ----------
    input_video : str
        Path to the input video file.
    output_video_path : str
        Path where the processed video will be saved.
    resize_dim : int, optional
        Maximum dimension (width or height) to resize frames to (default=600).
    fps : float, optional
        Frame rate for the output video (default=10.0).
    """
    vc = cv2.VideoCapture(input_video)
    ret, first_frame = vc.read()
    if not ret:
        raise IOError(f"Cannot read first frame from {input_video}")

    scale = resize_dim / max(first_frame.shape[:2])
    first_frame = cv2.resize(first_frame, None, fx=scale, fy=scale)
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    mask = np.zeros_like(first_frame)
    mask[..., 1] = 255

    height, width = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while True:
        ret, frame = vc.read()
        if not ret:
            break

        frame = cv2.resize(frame, None, fx=scale, fy=scale)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                            pyr_scale=0.5, levels=5,
                                            winsize=11, iterations=5,
                                            poly_n=5, poly_sigma=1.1, flags=0)

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mask[..., 0] = ang * 180 / np.pi / 2
        mask[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        rgb_flow = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
        output_frame = cv2.addWeighted(frame, 1, rgb_flow, 2, 0)

        out.write(output_frame)
        prev_gray = gray

    vc.release()
    out.release()
    cv2.destroyAllWindows()


'''
def compute_and_save_optical_flow(image1_path, image2_path,
                                  output_flow_path="dense_optical_flow.png"):
    """
    Reads two image files, computes the optical flow between them,
    and saves the optical flow visualization.

    Parameters:
    - image1_path (str): File path to the first image.
    - image2_path (str): File path to the second image.
    - output_flow_path (str): File path to save the optical flow visualization.
      Make sure this path has a valid image extension (e.g., .png, .jpg).
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_flow_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the two image frames
    frame1 = cv.imread(image1_path)
    frame2 = cv.imread(image2_path)

    # Check if the images are loaded properly
    if frame1 is None or frame2 is None:
        print("Error: One or both image files could not be loaded.")
        return

    # Convert images to grayscale
    prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    nxt = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

    # Prepare an HSV image for visualization; set saturation to maximum (255)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255

    # Compute the optical flow using Farneback's method
    flow = cv.calcOpticalFlowFarneback(prvs, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Convert flow vectors to magnitude and angle
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])

    # Map the angle (direction) to the hue channel (scaled to [0,180])
    hsv[..., 0] = ang * 180 / np.pi / 2
    # Normalize the magnitude and map it to the value channel
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)

    # Convert the HSV image to BGR color space for visualization
    bgr_flow = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    # Save only the optical flow visualization
    cv.imwrite(output_flow_path, bgr_flow)

    print(f"Optical flow visualization saved as '{output_flow_path}'")

'''

compute_dense_optical_flow(
    input_video="drone_footage/cut1.mp4",
    output_video_path="output/dense_optical_flow/dense_flow_output.mp4",
    resize_dim=600,
    fps=1
)
