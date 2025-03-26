import os, sys, argparse
import cv2
import torch
import numpy as np

# Paths — adjust as needed
RAFT_ROOT    = os.path.join(os.path.dirname(__file__), "RAFT")
MODEL_PATH   = os.path.join(RAFT_ROOT, "models", "raft-things.pth")
VIDEO_PATH   = "drone_footage/cut1.mp4"
OUTPUT_DIR   = "output/raft_optical_flow"
OUTPUT_FILE  = os.path.join(OUTPUT_DIR, "raft_flow.png")

# Insert RAFT into Python path
sys.path.insert(0, RAFT_ROOT)
sys.path.insert(0, os.path.join(RAFT_ROOT, "core"))
from raft import RAFT
from utils.utils import InputPadder

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load video frames
cap = cv2.VideoCapture(VIDEO_PATH)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.set(cv2.CAP_PROP_POS_FRAMES, 4)
_, frame1 = cap.read()
cap.set(cv2.CAP_PROP_POS_FRAMES, 10)
_, frame2 = cap.read()
cap.release()

# Initialize RAFT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = argparse.Namespace(small=False, mixed_precision=False, alternate_corr=False)
model = RAFT(args).to(device).eval()
ckpt = torch.load(MODEL_PATH, map_location=device)
state = {k.replace("module.",""):v for k,v in ckpt.items()}
model.load_state_dict(state)

def compute_raft_flow(im1, im2):
    t1 = torch.from_numpy(im1[..., ::-1].copy()).permute(2,0,1).unsqueeze(0).float()/255
    t2 = torch.from_numpy(im2[..., ::-1].copy()).permute(2,0,1).unsqueeze(0).float()/255
    padder = InputPadder(t1.shape)
    t1, t2 = padder.pad(t1.to(device), t2.to(device))
    with torch.no_grad():
        _, flow = model(t1, t2, iters=20, test_mode=True)
    return padder.unpad(flow[0]).permute(1,2,0).cpu().numpy()

def flow_to_color(flow):
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1], angleInDegrees=True)
    hsv = np.zeros((*flow.shape[:2],3), dtype=np.uint8)
    hsv[...,1] = 255
    hsv[...,0] = (ang/2).astype(np.uint8)
    hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# Compute and save
flow = compute_raft_flow(frame1, frame2)
color = flow_to_color(flow)
cv2.imwrite(OUTPUT_FILE, color)
print("Saved RAFT flow visualization to", OUTPUT_FILE)
