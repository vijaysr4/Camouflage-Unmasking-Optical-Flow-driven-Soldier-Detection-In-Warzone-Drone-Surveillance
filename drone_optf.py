import cv2 as cv
import numpy as np
import os

# Path to your video
VIDEO_PATH = "drone_footage/cut1.mp4"

# Output directory (relative)
OUTPUT_DIR = "./out_dro"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Open video
cap = cv.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Cannot open video: {VIDEO_PATH}")
    exit()

# Get total number of frames in the video
num_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
print(f"Total frames in video: {num_frames}")

# We need at least 10 frames to safely grab the 5th and (last - 5)
if num_frames < 10:
    print("Video doesn't have enough frames to get 5th and last-5 frames.")
    cap.release()
    exit()

# 1-based frame indexing:
#   5th frame -> index 4 in 0-based
#   (last - 5)th frame -> index (num_frames - 5) - 1 = num_frames - 6 in 0-based

frame_index_5th = 4
frame_index_lastminus5 = 10

# Seek and read the 5th frame
cap.set(cv.CAP_PROP_POS_FRAMES, frame_index_5th)
ret1, frame1 = cap.read()
if not ret1:
    print(f"Could not read frame at index {frame_index_5th}")
    cap.release()
    exit()

# Seek and read the (last - 5)th frame
cap.set(cv.CAP_PROP_POS_FRAMES, frame_index_lastminus5)
ret2, frame2 = cap.read()
if not ret2:
    print(f"Could not read frame at index {frame_index_lastminus5}")
    cap.release()
    exit()

cap.release()

# Convert both frames to grayscale
gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

# Calculate optical flow (Farneback) between the two frames
flow = cv.calcOpticalFlowFarneback(
    gray1, gray2,
    None,
    0.5,  # pyr_scale
    3,    # levels
    15,   # winsize
    3,    # iterations
    5,    # poly_n
    1.2,  # poly_sigma
    0     # flags
)

# Create an HSV visualization of the flow
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
hsv[..., 0] = ang * 180 / np.pi / 2
hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
bgr_flow = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

# Save the flow visualization
output_file = os.path.join(OUTPUT_DIR, "optflow_5th_to_lastminus5.png")
cv.imwrite(output_file, bgr_flow)
print(f"Optical flow computed and saved to {output_file}")
