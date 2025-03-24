import sys
import os

# Add the SPyNet repository inner package directory to the system path
current_dir = os.path.dirname(os.path.abspath(__file__))
# Assuming the pytorch-spynet repo is in a folder named "pytorch-spynet" in the current directory.
spynet_pkg_dir = os.path.join(current_dir, "pytorch-spynet", "spynet")
sys.path.insert(0, spynet_pkg_dir)

import cv2 as cv
import numpy as np
import torch

# Import the SPyNet model from the models folder.
from models.spynet import SPyNet

#############################################
# SPyNet Inference Class and Helper Function
#############################################

class SPyNetInference:
    def __init__(self, device='cuda'):
        """
        Initializes SPyNet with default pretrained weights.
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = SPyNet().to(self.device)
        self.model.eval()

    def predict(self, img1_bgr, img2_bgr):
        """
        Computes the optical flow between two BGR images using SPyNet.
        Returns:
          flow: A NumPy array of shape [H, W, 2] with the estimated flow.
        """
        # Convert BGR -> RGB, HWC -> CHW, and normalize to [0, 1]
        img1_rgb = cv.cvtColor(img1_bgr, cv.COLOR_BGR2RGB)
        img2_rgb = cv.cvtColor(img2_bgr, cv.COLOR_BGR2RGB)
        img1 = torch.from_numpy(img1_rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(self.device)
        img2 = torch.from_numpy(img2_rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # SPyNet returns a flow tensor of shape [B, 2, H, W]
            flow = self.model(img1, img2)
        # Remove batch dimension and convert to HWC format
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
# Main Script: Optical Flow on Two Input Images
#############################################

# Define image paths
image1_path = "extracted_frames/frame_0005.jpg"
image2_path = "extracted_frames/frame_0011.jpg"

# Load images
img1 = cv.imread(image1_path)
img2 = cv.imread(image2_path)

if img1 is None or img2 is None:
    print("Error: One or both images could not be loaded.")
    exit(1)

# Initialize SPyNet
spynet_infer = SPyNetInference(device='cuda')

# Compute SPyNet optical flow between the two images
flow = spynet_infer.predict(img1, img2)

# Convert the computed flow to a color image for visualization
colored_flow = flow_to_color(flow)

# Define output directory and save the result
output_dir = "output/pytorch_spynet"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "optflow_spynet.png")
cv.imwrite(output_file, colored_flow)
print("SPyNet optical flow computed and saved to", output_file)
