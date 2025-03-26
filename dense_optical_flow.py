import os
import cv2
import numpy as np


def compute_dense_optical_flow(image1_path: str, image2_path: str) -> np.ndarray:
    """
    Compute dense optical flow (Farneback) between two images and return a HxWx2 float32 flow array.

    Parameters:
        image1_path (str): Path to the first frame.
        image2_path (str): Path to the second frame.

    Returns:
        np.ndarray: Optical flow array of shape (height, width, 2).
    """
    # Load frames
    frame1 = cv2.imread(image1_path)
    frame2 = cv2.imread(image2_path)
    if frame1 is None or frame2 is None:
        raise FileNotFoundError(f"Could not load images '{image1_path}' or '{image2_path}'")

    # Convert to grayscale
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    nxt  = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Compute Farneback flow
    flow = cv2.calcOpticalFlowFarneback(prvs, nxt, None,
                                        pyr_scale=0.5, levels=3,
                                        winsize=15, iterations=3,
                                        poly_n=5, poly_sigma=1.2,
                                        flags=0)
    return flow


def visualize_flow(flow: np.ndarray, output_path: str) -> None:
    """Save HSV visualization of a dense flow array as an image."""
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, bgr)
    print(f"Saved visualization: {output_path}")


# Example usage
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Compute dense optical flow between two images.")
    parser.add_argument('img1')
    parser.add_argument('img2')
    parser.add_argument('--vis', '-v', help='Path to save HSV visualization (optional)')
    args = parser.parse_args()

    flow = compute_dense_optical_flow(args.img1, args.img2)
    print(f"Computed flow with shape {flow.shape}")

    if args.vis:
        visualize_flow(flow, args.vis)
