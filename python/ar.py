import numpy as np
import cv2
from tqdm import tqdm
from opts import get_opts

#Import necessary functions

from helper import loadVid

def crop_save_video(ar_src_vid_path: str, ar_img_path: str) -> None:
    '''
    Function to crop the video to the center of each frame.add()
    The cropped video's ratio will be the same as the reference image. (cv_cover.jpg)
    The cropped video will be saved as '../data/cropped_ar_source.mov'.
    '''
    input_video = loadVid(ar_src_vid_path) # shape: (num_frames, height, width, 3)
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
    out = cv2.VideoWriter('../data/cropped_ar_source.mov', fourcc, fps, (new_width, new_height))

    for frame in tqdm(cropped_video):
        out.write(frame)

    print(f"Video saved successfully at '../data/cropped_ar_source.mov'")
    out.release()


if __name__ == "__main__":
    opts = get_opts()

    if opts.crop_video:
        crop_save_video(opts.ar_src_vid_path, opts.ar_img_path)