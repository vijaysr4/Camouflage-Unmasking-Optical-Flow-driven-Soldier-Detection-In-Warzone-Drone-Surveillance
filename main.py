import os
from pathlib import Path

import vid_preprocessing
import object_detect
import pytorch_spynet


def main():
    # 1) Extract frames from the MP4 video
    input_video = "drone_footage/cut1.mp4"
    frames_output = "output/main/frames"  # subfolder for frames
    print(f"Extracting frames 4..8 from {input_video} into '{frames_output}'...")
    os.makedirs(frames_output, exist_ok=True)

    vid_preprocessing.extract_frames(
        vid=input_video,
        f_start=4,
        f_end=8,
        interval=None,        # extract only frames 4 and 8 (start + end) if interval=None
        output_folder=frames_output
    )

    # Run object detection on the entire MP4 video
    print(f"Running object detection on {input_video} ...")
    # If 'object_detect.py' is set to use 'cut1.mp4' internally, just call its main():
    object_detect.main()

    # Compute SPyNet optical flow between frame_0004.jpg and frame_0008.jpg
    print("Computing SPyNet optical flow between frames 4 and 8...")
    spynet_output = "output/main/pytorch_spynet"
    os.makedirs(spynet_output, exist_ok=True)

    frame1_path = os.path.join(frames_output, "frame_0004.jpg")
    frame2_path = os.path.join(frames_output, "frame_0008.jpg")

    if not (os.path.isfile(frame1_path) and os.path.isfile(frame2_path)):
        print(f"Missing {frame1_path} or {frame2_path}. Cannot run SPyNet.")
        return

    flow, colored = pytorch_spynet.compute_spynet_flow(
        img1_path=frame1_path,
        img2_path=frame2_path,
        output_dir=spynet_output
    )
    print("Optical flow shape:", flow.shape)
    print(f"SPyNet .flo and color map saved in '{spynet_output}'.")


if __name__ == "__main__":
    main()
