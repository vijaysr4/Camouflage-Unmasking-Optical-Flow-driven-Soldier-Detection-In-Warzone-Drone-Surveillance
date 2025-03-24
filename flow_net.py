import os
import cv2
import torch
import numpy as np

##############################################################################
# 1. FlowNet2 Imports (adjust if your installed version differs)
##############################################################################
try:
    from flownet2_pytorch.models import FlowNet2  # or from flownet2_pytorch import FlowNet2
except ImportError:
    print("ERROR: FlowNet2 not found. Make sure you've installed flownet2-pytorch correctly.")
    exit()

##############################################################################
# 2. Paths and Settings
##############################################################################
VIDEO_PATH = "drone_footage/cut1.mp4"               # Path to your video
CHECKPOINT_PATH = "FlowNet2_checkpoint.pth.tar"     # Pretrained checkpoint
OUTPUT_DIR = "./out_dro"
os.makedirs(OUTPUT_DIR, exist_ok=True)

##############################################################################
# 3. Read the 5th Frame and (Last−5)th Frame from the Video
##############################################################################
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Cannot open video: {VIDEO_PATH}")
    exit()

num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total frames in video: {num_frames}")

# We'll assume the video has >= 10 frames
if num_frames < 10:
    print("Video doesn't have enough frames to pick the 5th and (last-5)th.")
    cap.release()
    exit()

frame_index_5th = 4              # 0-based index for 5th frame
frame_index_last_minus_5 = num_frames - 6

# Seek & read the 5th frame
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index_5th)
ret1, frame1 = cap.read()
if not ret1:
    print(f"Could not read the 5th frame at index {frame_index_5th}.")
    cap.release()
    exit()

# Seek & read the (last−5)th frame
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index_last_minus_5)
ret2, frame2 = cap.read()
if not ret2:
    print(f"Could not read the (last-5)th frame at index {frame_index_last_minus_5}.")
    cap.release()
    exit()

cap.release()

##############################################################################
# 4. Initialize FlowNet2 and Load Checkpoint
##############################################################################
class Args:
    """Dummy class to hold FlowNet2 settings."""
    fp16 = False
    rgb_max = 255.0  # FlowNet2 was trained with images scaled to [0..255]

args = Args()

# Instantiate the model
model = FlowNet2(args)

# Load the pretrained weights
checkpoint = torch.load(CHECKPOINT_PATH)
model.load_state_dict(checkpoint["state_dict"], strict=False)

# Send model to GPU and set to eval mode
model.cuda()
model.eval()

##############################################################################
# 5. Preprocess the Two Frames for FlowNet2
#
# FlowNet2 expects input as [N, 6, H, W], where the first 3 channels are the
# first image (RGB) and the next 3 channels are the second image (RGB).
##############################################################################
def preprocess_frame_for_flownet2(img_bgr):
    """
    Convert a BGR uint8 image (H,W,3) -> Float32 [3,H,W] in RGB order,
    retaining the [0..255] pixel range (FlowNet2 expects that range).
    """
    # BGR -> RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    # [H,W,3] -> [3,H,W]
    img_rgb = np.transpose(img_rgb, (2, 0, 1))
    return img_rgb

frame1_tensor = preprocess_frame_for_flownet2(frame1)  # shape [3,H,W]
frame2_tensor = preprocess_frame_for_flownet2(frame2)  # shape [3,H,W]

# Concatenate along channel dimension -> shape [6,H,W]
combined = np.concatenate((frame1_tensor, frame2_tensor), axis=0)
combined_torch = torch.from_numpy(combined).unsqueeze(0).cuda()  # [1,6,H,W]

##############################################################################
# 6. Forward Pass: Compute Optical Flow
##############################################################################
with torch.no_grad():
    flow_output = model(combined_torch)  # shape [1,2,H,W]
flow = flow_output.squeeze(0).cpu().numpy()  # -> [2,H,W]

flow_x = flow[0, :, :]
flow_y = flow[1, :, :]

##############################################################################
# 7. Visualize Flow in HSV (similar to Farneback style)
##############################################################################
mag, ang = cv2.cartToPolar(flow_x, flow_y, angleInDegrees=False)

# Create an HSV image
h, w = flow_x.shape
hsv = np.zeros((h, w, 3), dtype=np.uint8)

# Hue: (angle in radians) -> degrees -> scale to [0..180]
hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
# Saturation
hsv[..., 1] = 255
# Value: magnitude normalized to [0..255]
mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
hsv[..., 2] = mag_norm.astype(np.uint8)

# Convert HSV -> BGR for saving
bgr_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

##############################################################################
# 8. Save Results
##############################################################################
output_file = os.path.join(OUTPUT_DIR, "flownet2_optflow_5th_to_lastminus5.png")
cv2.imwrite(output_file, bgr_flow)
print(f"FlowNet2 optical flow visualization saved to {output_file}")
