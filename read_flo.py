import numpy as np
import cv2 as cv

def read_flo_file(filename):
    """Read .flo optical flow file in Middlebury format."""
    with open(filename, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)[0]
        if magic != 202021.25:
            raise ValueError("Magic number incorrect. Invalid .flo file")
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        # Read the flow data (2 channels: horizontal and vertical)
        data = np.fromfile(f, np.float32, count=int(2 * w * h))
        flow = np.resize(data, (h, w, 2))
    return flow

def flow_to_color(flow):
    """Convert optical flow into a BGR image for visualization."""
    h, w, _ = flow.shape
    flow_x = flow[..., 0]
    flow_y = flow[..., 1]
    mag, ang = cv.cartToPolar(flow_x, flow_y, angleInDegrees=True)
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 1] = 255
    hsv[..., 0] = (ang / 2).astype(np.uint8)  # Scale angle to [0,180)
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    return bgr

# Usage example:
flo_file = "output/pytorch_spynet/out.flo"
flow = read_flo_file(flo_file)
colored_flow = flow_to_color(flow)
cv.imwrite("output/pytorch_spynet/out_color.png", colored_flow)
print("Saved color visualization to output/pytorch_spynet/out_color.png")
