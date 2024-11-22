import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import LucasKanade
from tqdm import tqdm

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument(
    '--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade'
)
parser.add_argument(
    '--threshold',
    type=float,
    default=1e-2,
    help='dp threshold of Lucas-Kanade for terminating optimization',
)
parser.add_argument(
    '--template_threshold',
    type=float,
    default=5,
    help='threshold for determining whether to update template',
)
args = parser.parse_args()
num_iters = int(args.num_iters)
threshold = args.threshold
template_threshold = args.template_threshold

seq = np.load("../data/girlseq.npy")
rect = [280, 152, 330, 318]
rects = []

# The reference template we maintain
T_frame = seq[:, :, 0]
T_rect = rect.copy()

# Keep the first template for drift correction
T1 = seq[:, :, 0]
T1_rect = rect.copy()
p_agg = np.zeros(2) # The transformation aggregate from the first frame

for i in tqdm(range(seq.shape[2] - 1)):
    It = seq[:, :, i] # current
    It1 = seq[:, :, i + 1] # next

    # Step 1: Perform normal Lucas-Kanade tracking with the current template
    p = LucasKanade(T_frame, It1, T_rect, threshold, num_iters)
    pn = p_agg + p

    # Step 2: Perform drift correction with the first template
    p_star = LucasKanade(T1, It1, T1_rect, threshold, num_iters, p0=p_agg)
    
    # If the drift is small, update the template
    if np.linalg.norm(p_star - pn) <= template_threshold:
        # Update the template
        T_frame = It1.copy()
        T_rect[0] = T1_rect[0] + p_star[0] # x1
        T_rect[1] = T1_rect[1] + p_star[1] # y1
        T_rect[2] = T1_rect[2] + p_star[0] # x2
        T_rect[3] = T1_rect[3] + p_star[1] # y2

        p_agg = p_star
        # p_agg = pn

    rects.append(T_rect.copy())

# (num_frames, 4)
np.save("girlseqrects-wcrt.npy", rects)