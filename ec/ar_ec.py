import numpy as np
import cv2
from loadVid import loadVid
import sys
import os
import time

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../python')))

from python.opts import get_opts
from python.planarH import compositeH

# Q3.2
def warpImage(opts, template_img, from_img, to_img):
    '''
    Function to warp from_img onto to_img using homography computed between template_img and to_img.
    Return the composite image after warping the from_img on top of to_img using the homography.
    '''
    # Resize from_img such that the warpped image fit the correct size as template_img
    from_img = cv2.resize(from_img, (template_img.shape[1], template_img.shape[0]))

    # Use cv2 ORB to match features
    orb = cv2.ORB_create(nfeatures=2000)
    kp1, des1 = orb.detectAndCompute(template_img, None)
    kp2, des2 = orb.detectAndCompute(to_img, None)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)
    
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
            
    if len(good_matches) < 4:
        print("Not enough matches found!", file=sys.stderr)
        return None
    
    locs1 = np.array([kp1[m.queryIdx].pt for m in good_matches])
    locs2 = np.array([kp2[m.trainIdx].pt for m in good_matches])
    
    H, _ = cv2.findHomography(locs1, locs2, cv2.RANSAC, opts.inlier_tol)
    
    composite_img = compositeH(H, from_img, to_img)

    return composite_img


def cropFrame(frame, ratio):
    '''
    Function to crop a frame to the center based on the ratio of the reference image.
    Return the cropped frame.
    '''
    vid_height, vid_width = frame.shape[:2] # (360, 640)
    vid_ratio = vid_width / vid_height

    if vid_ratio > ratio:
        # Crop horizontally
        new_width = int(vid_height * ratio) # 286
        new_height = vid_height # 360
    else:
        # Crop vertically
        new_width = vid_width
        new_height = int(vid_width / ratio)
    
    start_x = (vid_width - new_width) // 2
    start_y = (vid_height - new_height) // 2

    cropped_frame = frame[start_y:start_y+new_height, start_x:start_x+new_width, :]
    return cropped_frame


def calculate_fps(start_time, frame_count):
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    return fps


if __name__ == "__main__":
    opts = get_opts()

    # Load the source video
    source_video = loadVid(opts.ar_src_vid_path)
    print(f"Loaded source videos at {opts.ar_src_vid_path}, shape: {source_video.shape}") # (511, 360, 640, 3)

    # Load the target video
    target_video = loadVid(opts.ar_tgt_vid_path)
    print(f"Loaded target videos at {opts.ar_tgt_vid_path}, shape: {target_video.shape}") # (641, 480, 640, 3)

    # Load the reference image
    ref_image = cv2.imread(opts.ar_img_path)
    print(f"Loaded reference image at {opts.ar_img_path}, shape: {ref_image.shape}") # (440, 350, 3)
    
    ref_height, ref_width = ref_image.shape[:2] # (440, 350)
    ref_ratio = ref_width / ref_height

    # Initialize variables for FPS calculation
    n_frames_sec = 0
    start_time = time.time()

    # Generate warpped images and use cv2.imshow to show each frames
    result_frame = min(source_video.shape[0], target_video.shape[0])
    for i in range(result_frame):
        cropped_frame = cropFrame(source_video[i], ref_ratio)
        composite_img = warpImage(opts, ref_image, cropped_frame, target_video[i])

        n_frames_sec += 1
        fps = calculate_fps(start_time, n_frames_sec)
        # Reset FPS calculation every 1 second
        if time.time() - start_time > 1:
            n_frames_sec = 0
            start_time = time.time()

        # Display FPS
        cv2.putText(composite_img, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("AR Video", composite_img)
        cv2.waitKey(1)

