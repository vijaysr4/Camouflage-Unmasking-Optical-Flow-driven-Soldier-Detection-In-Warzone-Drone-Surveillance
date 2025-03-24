import os
import math
import numpy as np
import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm  # pip install tqdm


class PositionalEncoding2D(nn.Module):
    """
    2D Positional Encoding for a feature map of shape (B, C, H, W).
    Adds sine/cosine positional encodings over spatial dimensions.
    """

    def __init__(self, d_model):
        super(PositionalEncoding2D, self).__init__()
        self.d_model = d_model

    def forward(self, x):
        # x: (B, C, H, W) with C == d_model
        B, C, H, W = x.size()
        pe = torch.zeros(B, C, H, W, device=x.device)

        y_pos = torch.arange(0, H, device=x.device, dtype=torch.float32).unsqueeze(1).repeat(1, W)
        x_pos = torch.arange(0, W, device=x.device, dtype=torch.float32).unsqueeze(0).repeat(H, 1)

        div_term = torch.exp(torch.arange(0, C, 2, device=x.device, dtype=torch.float32) *
                             -(math.log(10000.0) / C))
        for i in range(0, C, 2):
            pe[:, i, :, :] = torch.sin(x_pos * div_term[i // 2])
            if i + 1 < C:
                pe[:, i + 1, :, :] = torch.cos(y_pos * div_term[i // 2])
        return x + pe


class FlowRefiner(nn.Module):
    """
    Improved flow refiner that projects the 2-channel flow into a higher-dimensional space,
    adds positional encoding, processes it with a Transformer, and then projects back to 2 channels.

    To reduce memory usage, the flow is downsampled before processing.
    """

    def __init__(self, d_model=64, nhead=8, num_layers=4, downscale_factor=16):
        super(FlowRefiner, self).__init__()
        self.downscale_factor = downscale_factor
        # Project 2-channel flow into d_model channels.
        self.input_proj = nn.Conv2d(2, d_model, kernel_size=1)
        self.pos_encoding = PositionalEncoding2D(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Project back from d_model channels to 2 channels.
        self.output_proj = nn.Conv2d(d_model, 2, kernel_size=1)

    def forward(self, flow):
        # flow: (H, W, 2) numpy array.
        device = next(self.parameters()).device
        H, W, _ = flow.shape

        # Downsample flow to reduce token count.
        new_H, new_W = H // self.downscale_factor, W // self.downscale_factor
        flow_ds = np.stack([cv.resize(flow[..., i], (new_W, new_H), interpolation=cv.INTER_LINEAR)
                            for i in range(2)], axis=-1)

        # Convert to tensor: (1, 2, new_H, new_W)
        x = torch.from_numpy(flow_ds).float().to(device).permute(2, 0, 1).unsqueeze(0)

        # Project to d_model channels.
        x = self.input_proj(x)  # (1, d_model, new_H, new_W)
        # Add positional encoding.
        x = self.pos_encoding(x)  # (1, d_model, new_H, new_W)
        # Flatten spatial dimensions: (1, new_H*new_W, d_model)
        x = x.flatten(2).transpose(1, 2)
        # Process with transformer.
        x = self.transformer_encoder(x)  # (1, new_H*new_W, d_model)
        # Reshape back: (1, d_model, new_H, new_W)
        x = x.transpose(1, 2).reshape(1, -1, new_H, new_W)
        # Project back to 2 channels.
        x = self.output_proj(x)  # (1, 2, new_H, new_W)
        # Upsample back to original resolution.
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        refined_flow = x.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        return refined_flow


def compute_and_save_optical_flow_with_transformer(image1_path, image2_path,
                                                   output_flow_path="dense_optical_flow_transformer.png",
                                                   downscale_factor=16):
    """
    Reads two images, computes dense optical flow using Farneback,
    refines the flow using the improved transformer module (with downsampling),
    and saves the visualization.
    """
    # Ensure output directory exists.
    output_dir = os.path.dirname(output_flow_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load images.
    frame1 = cv.imread(image1_path)
    frame2 = cv.imread(image2_path)
    if frame1 is None or frame2 is None:
        print("Error: One or both image files could not be loaded.")
        return

    # Convert images to grayscale.
    prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    nxt = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

    # Compute dense optical flow using Farneback.
    flow = cv.calcOpticalFlowFarneback(prvs, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Initialize the transformer refiner with increased downscale factor.
    refiner = FlowRefiner(d_model=64, nhead=8, num_layers=4, downscale_factor=downscale_factor)
    refiner.eval()  # evaluation mode.

    # Move model to GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    refiner.to(device)

    with torch.no_grad():
        refined_flow = refiner(flow)

    # Prepare HSV image for visualization.
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255

    # Compute magnitude and angle from refined flow.
    mag, ang = cv.cartToPolar(refined_flow[..., 0], refined_flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)

    # Convert HSV to BGR.
    bgr_flow = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    # Save visualization.
    cv.imwrite(output_flow_path, bgr_flow)
    print(f"Improved optical flow visualization saved as '{output_flow_path}'")


# Example usage:
compute_and_save_optical_flow_with_transformer("extracted_frames/frame_0005.jpg",
                                               "extracted_frames/frame_0011.jpg",
                                               output_flow_path="output/modified_dof/dense_optical_flow_transformer.png",
                                               downscale_factor=16)
