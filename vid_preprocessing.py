import cv2
import os


def extract_frames(vid, f_start, f_end, interval=None, output_folder='frames'):
    """
    Extract frames from a video between f_start and f_end.

    Parameters:
    - vid (str): Path to the input video file.
    - f_start (int): Starting frame number.
    - f_end (int): Ending frame number.
    - interval (int or None): If provided, save every 'interval'-th frame from f_start to f_end.
                              If None, only save f_start and f_end frames.
    - output_folder (str): Folder where extracted frames will be saved.
    """
    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(vid)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f_end = min(f_end, total_frames - 1)  # Ensure f_end doesn't exceed the video length

    # Determine which frames to extract
    if interval is not None:
        frames_to_extract = list(range(f_start, f_end + 1, interval))
    else:
        frames_to_extract = [f_start, f_end]

    extracted_count = 0
    current_frame = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        if current_frame in frames_to_extract:
            output_path = os.path.join(output_folder, f"frame_{current_frame:04d}.jpg")
            cv2.imwrite(output_path, frame)
            extracted_count += 1

        current_frame += 1
        if current_frame > f_end:
            break  # No need to process beyond the ending frame

    cap.release()
    print(f"Extracted {extracted_count} frame(s) to the folder '{output_folder}'.")

# Example usage:
extract_frames("drone_footage/cut1.mp4", 4, 8, interval=None, output_folder='extracted_frames')
# extract_frames("input_video.mp4", 50, 150, interval=None)
