import numpy as np
import cv2 as cv
import os


def compute_and_save_lk_optical_flow_for_images(image1_path, image2_path, output_image_path):
    """
    Reads two image files, computes Lucas-Kanade optical flow to track features from the first image to the second,
    draws the motion tracks on the second image, and saves the visualization to the specified output path.

    Parameters:
    - image1_path (str): Path to the first image.
    - image2_path (str): Path to the second image.
    - output_image_path (str): File path to save the output visualization image.
      Make sure to include a valid image extension (e.g., .png, .jpg).
    """
    # Load the two images
    img1 = cv.imread(image1_path)
    img2 = cv.imread(image2_path)

    # Check if images were loaded correctly
    if img1 is None or img2 is None:
        print("Error: Could not load one or both images.")
        return

    # Convert images to grayscale
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    # Parameters for Shi-Tomasi corner detection
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)

    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    # Detect features (corners) in the first image
    p0 = cv.goodFeaturesToTrack(gray1, mask=None, **feature_params)
    if p0 is None:
        print("No features found in the first image.")
        return

    # Calculate optical flow from image1 to image2 using Lucas-Kanade method
    p1, st, err = cv.calcOpticalFlowPyrLK(gray1, gray2, p0, None, **lk_params)
    if p1 is None or st is None:
        print("Error: Optical flow could not be computed.")
        return

    # Select good points that were successfully tracked
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # Create a mask image for drawing the optical flow tracks
    mask = np.zeros_like(img1)
    # Generate random colors for each track
    color = np.random.randint(0, 255, (100, 3))

    # Draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        img2 = cv.circle(img2, (int(a), int(b)), 5, color[i].tolist(), -1)

    # Overlay the tracks on the second image
    output_img = cv.add(img2, mask)

    # Ensure the output directory exists and save the output image
    output_dir = os.path.dirname(output_image_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cv.imwrite(output_image_path, output_img)
    print(f"Optical flow visualization saved as '{output_image_path}'")

# Example usage:
compute_and_save_lk_optical_flow_for_images("extracted_frames/frame_0004.jpg",
                                            "extracted_frames/frame_0011.jpg",
                                            "output/LK_optical_flow/lk_optical_flow.png")
