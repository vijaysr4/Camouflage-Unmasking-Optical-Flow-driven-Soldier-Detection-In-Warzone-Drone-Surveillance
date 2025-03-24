import sys
import os

# Add the SPyNet repository directory to the system path
current_dir = os.path.dirname(os.path.abspath(__file__))
spynet_repo_dir = os.path.join(current_dir, "spynet")
sys.path.insert(0, spynet_repo_dir)

import cv2 as cv
import numpy as np
import torch

# Import SPyNet model from the spynet repo.
# With sys.path pointing to the spynet directory, the models folder is at the top level.
from models.spynet import SPyNet

#############################################
# SPyNet Inference Class and Helper Function
#############################################

class SPyNetInference:
    def __init__(self, model_path, device='cuda'):
        """
        Initializes SPyNet with the given pretrained weights.
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = SPyNet()
        # Load the pretrained weights
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, img1_bgr, img2_bgr):
        """
        Computes the optical flow between two BGR images using SPyNet.
        Returns:
          flow: A NumPy array of shape [H, W, 2] with the estimated flow.
        """
        # Convert BGR -> RGB, HWC -> CHW, and normalize to [0, 1]
        img1 = torch.from_numpy(img1_bgr[:, :, ::-1].copy()).float().div(255.0)
        img2 = torch.from_numpy(img2_bgr[:, :, ::-1].copy()).float().div(255.0)
        # Add batch dimension and change shape from HWC to CHW
        img1 = img1.unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
        img2 = img2.unsqueeze(0).permute(0, 3, 1, 2).to(self.device)

        with torch.no_grad():
            # SPyNet takes two images and returns a flow tensor.
            # Expected output shape: [B, 2, H, W]
            flow = self.model(img1, img2)
        # Remove batch dimension and change to HWC
        flow = flow[0].permute(1, 2, 0).cpu().numpy()
        return flow

def flow_to_color(flow):
    """
    Converts an optical flow map (H x W x 2) into a BGR color image using HSV color coding.
      - Hue corresponds to the flow direction.
      - Value corresponds to the flow magnitude.
    """
    h, w, _ = flow.shape
    flow_x = flow[..., 0]
    flow_y = flow[..., 1]

    # Compute magnitude and angle of flow vectors
    mag, ang = cv.cartToPolar(flow_x, flow_y, angleInDegrees=True)

    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 1] = 255
    hsv[..., 0] = (ang / 2).astype(np.uint8)  # Scale angle to [0,180)
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    return bgr

#############################################
# Main Script: Video Frame Extraction & SPyNet Optical Flow
#############################################

# Path to your input video
VIDEO_PATH = "drone_footage/cut1.mp4"  # Update with your video file

# Output directory for saving the flow visualization
OUTPUT_DIR = "./out_spynet"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Open the video file
cap = cv.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Cannot open video:", VIDEO_PATH)
    exit()

# Get total number of frames in the video
num_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
print("Total frames in video:", num_frames)
if num_frames < 10:
    print("Video doesn't have enough frames to process.")
    cap.release()
    exit()

# Choose frame indices (0-based); here we use the 5th and the 10th frame
frame_index_5th = 4
frame_index_lastminus5 = 10

# Read the 5th frame
cap.set(cv.CAP_PROP_POS_FRAMES, frame_index_5th)
ret1, frame1 = cap.read()
if not ret1:
    print(f"Could not read frame at index {frame_index_5th}")
    cap.release()
    exit()

# Read the 10th frame
cap.set(cv.CAP_PROP_POS_FRAMES, frame_index_lastminus5)
ret2, frame2 = cap.read()
if not ret2:
    print(f"Could not read frame at index {frame_index_lastminus5}")
    cap.release()
    exit()

cap.release()

# Path to the pretrained SPyNet weights (update this path as needed)
MODEL_PATH = "spynet/weights/SpyNet_Final.pth"

# Initialize SPyNet
spynet_infer = SPyNetInference(MODEL_PATH, device='cuda')

# Compute SPyNet optical flow between the two frames
flow = spynet_infer.predict(frame1, frame2)

# Convert the computed flow to a color image for visualization
bgr_flow = flow_to_color(flow)

# Save the resulting optical flow visualization image
output_file = os.path.join(OUTPUT_DIR, "optflow_spynet.png")
cv.imwrite(output_file, bgr_flow)
print("SPyNet optical flow computed and saved to", output_file)
