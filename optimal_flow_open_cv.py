import numpy as np
import cv2 as cv
import os
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Dense Optical Flow on an image sequence from MPI-Sintel.')
parser.add_argument('sequence', type=str, help='Path to MPI-Sintel sequence folder')
args = parser.parse_args()

# Get sorted list of image files from the sequence folder (assumes .png images)
img_files = sorted([os.path.join(args.sequence, f)
                    for f in os.listdir(args.sequence) if f.endswith('.png')])

if len(img_files) < 2:
    print("Need at least two frames to compute optical flow.")
    exit(0)

# Create output folder if it doesn't exist
output_folder = "output_dense_optical_flow"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Read the first frame and convert to grayscale
prev_frame = cv.imread(img_files[0])
prev_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)

# Create an HSV image for flow visualization; copy the shape from the first frame
hsv = np.zeros_like(prev_frame)
hsv[..., 1] = 255

# Process each subsequent frame in the sequence
for i in range(1, len(img_files)):
    curr_frame = cv.imread(img_files[i])
    curr_gray = cv.cvtColor(curr_frame, cv.COLOR_BGR2GRAY)

    # Calculate dense optical flow using Farneback's algorithm
    flow = cv.calcOpticalFlowFarneback(prev_gray, curr_gray, None,
                                       0.5, 3, 15, 3, 5, 1.2, 0)

    # Compute magnitude and angle of the flow vectors
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])

    # Set HSV channels: Hue corresponds to flow direction, Value to flow magnitude
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)

    # Convert HSV to BGR (for visualization)
    bgr_flow = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    # Save the resulting flow image
    output_filename = os.path.join(output_folder, f'frame_{i:04d}.png')
    cv.imwrite(output_filename, bgr_flow)
    print(f"Saved {output_filename}")

    # Update previous frame for the next iteration
    prev_gray = curr_gray.copy()
