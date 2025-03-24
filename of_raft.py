import os
from glob import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from RAFT.core.raft import raft
from RAFT.core.utils import flow_viz
from RAFT.core.utils.utils import InputPadder
from RAFT.config import RAFTConfig

# Configure RAFT model
config = RAFTConfig(
    dropout=0,
    alternate_corr=False,
    small=False,
    mixed_precision=False
)

model = raft(config)
print(model)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# Specify the path to your pre-trained RAFT weights
weights_path = 'raft-sintel.pth'
ckpt = torch.load(weights_path, map_location=device)
model.to(device)
model.load_state_dict(ckpt)

# Define the path for the specific sample folder "ambush_5"
sample_folder = os.path.join('/Data/optimal_flow_cv/training/clean', 'ambush_5')

# Search for PNG images in the sample folder
image_files = sorted(glob(os.path.join(sample_folder, '*.png')))
print(f"Found {len(image_files)} images in '{sample_folder}':")
print(image_files)

# Use just the first two images
if len(image_files) < 2:
    raise ValueError("Not enough images in the sample folder to compute optical flow.")

# Select only the first two images
file1, file2 = image_files[0], image_files[1]
print(f"Processing image pair:\n{file1}\n{file2}")

def load_image(imfile, device):
    """Load an image file and convert it to a torch tensor."""
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(device)

def viz(img1, img2, flo):
    """Visualize two input images and the computed optical flow."""
    img1_np = img1[0].permute(1, 2, 0).cpu().numpy()
    img2_np = img2[0].permute(1, 2, 0).cpu().numpy()
    flo_np = flo[0].permute(1, 2, 0).cpu().numpy()

    # Convert flow to a colorful RGB image
    flo_rgb = flow_viz.flow_to_image(flo_np)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 4))
    ax1.set_title('Input Image 1')
    ax1.imshow(img1_np.astype(int))
    ax2.set_title('Input Image 2')
    ax2.imshow(img2_np.astype(int))
    ax3.set_title('Estimated Optical Flow')
    ax3.imshow(flo_rgb)
    plt.show()

# Load the two images
image1 = load_image(file1, device)
image2 = load_image(file2, device)

# Pad images to dimensions acceptable by the RAFT model
padder = InputPadder(image1.shape)
image1, image2 = padder.pad(image1, image2)

# Compute optical flow for the image pair
model.eval()
with torch.no_grad():
    flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)

# Visualize the results
viz(image1, image2, flow_up)
