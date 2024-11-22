import numpy as np
import cv2
from tqdm import tqdm
from opts import get_opts
from HarryPotterize import warpImage
import os
from helper import loadVid

def crop_save_video(ar_src_vid_path: str, ar_img_path: str, ar_crop_vid_path: str) -> None:
    '''
    Function to crop the video to the center of each frame.
    The cropped video's ratio will be the same as the reference image. (cv_cover.jpg)
    '''
    input_video = loadVid(ar_src_vid_path) # (num_frames, height, width, 3)
    vid_height, vid_width = input_video[0].shape[:2] # (360, 640)
    vid_ratio = vid_width / vid_height

    ref_image = cv2.imread(ar_img_path)
    ref_height, ref_width = ref_image.shape[:2] # (440, 350)
    ref_ratio = ref_width / ref_height

    if vid_ratio > ref_ratio: # This is our case
        # Crop horizontally
        new_width = int(vid_height * ref_ratio) # 286
        new_height = vid_height # 360
    else:
        # Crop vertically
        new_width = vid_width
        new_height = int(vid_width / ref_ratio)

    start_x = (vid_width - new_width) // 2
    start_y = (vid_height - new_height) // 2

    cropped_video = input_video[:, start_y:start_y+new_height, start_x:start_x+new_width, :]

    # Save the cropped video
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fps = 30 # ar_source.mov panda video has 30 fps
    out = cv2.VideoWriter(ar_crop_vid_path, fourcc, fps, (new_width, new_height))

    for frame in tqdm(cropped_video):
        out.write(frame)

    print(f"Video saved successfully to {ar_crop_vid_path}.")
    out.release()


if __name__ == "__main__":
    opts = get_opts()

    # Only first run should crop and save the video
    if opts.crop_video:
        crop_save_video(opts.ar_src_vid_path, opts.ar_img_path, opts.ar_crop_vid_path)

    # Create the output directory if it does not exist
    if not os.path.exists(opts.ar_out_dir):
        os.makedirs(opts.ar_out_dir)

    # Load the cropped video
    cropped_video = loadVid(opts.ar_crop_vid_path)
    print(f"Loaded cropped videos at {opts.ar_crop_vid_path}, shape: {cropped_video.shape}") # (511, 360, 286, 3)

    # Load the target video
    target_video = loadVid(opts.ar_tgt_vid_path)
    print(f"Loaded target videos at {opts.ar_tgt_vid_path}, shape: {target_video.shape}") # (641, 480, 640, 3)

    # Load the reference image
    ref_image = cv2.imread(opts.ar_img_path)
    print(f"Loaded reference image at {opts.ar_img_path}, shape: {ref_image.shape}") # (440, 350, 3)

    # Generate warpped images and save each frames
    result_frame = min(cropped_video.shape[0], target_video.shape[0])
    for i in tqdm(range(result_frame)): # 511 frames
        composite_img = warpImage(opts, ref_image, cropped_video[i], target_video[i])
        cv2.imwrite(f"{opts.ar_out_dir}frame_{i}.png", composite_img)

    # Reconstruct videos from saved frames
    out = cv2.VideoWriter(f"{opts.ar_out_dir}{opts.ar_out_vid}", cv2.VideoWriter_fourcc(*'MJPG'), 30, (640, 480))
    for i in tqdm(range(result_frame)):
        img = cv2.imread(f"{opts.ar_out_dir}frame_{i}.png")
        out.write(img)

    out.release()
    print(f"Video saved successfully at {opts.ar_out_dir}{opts.ar_out_vid}.")