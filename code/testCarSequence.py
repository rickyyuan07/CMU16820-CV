import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import LucasKanade
import os
from tqdm import tqdm

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
args = parser.parse_args()
num_iters = int(args.num_iters)
threshold = args.threshold

seq = np.load("../data/carseq.npy") # shape: (image_height, image_width, num_frames)
rect = [59, 116, 145, 151]
rects = []

for i in tqdm(range(seq.shape[2] - 1)):
    It = seq[:, :, i] # current
    It1 = seq[:, :, i + 1] # next

    p = LucasKanade(It, It1, rect, threshold, num_iters)
    rect[0] += p[0] # x1
    rect[1] += p[1] # y1
    rect[2] += p[0] # x2
    rect[3] += p[1] # y2

    rects.append(rect.copy())

    if not os.path.exists("../results"):
        os.makedirs("../results")

    # Display and save the image for frames 1, 100, 200, 300, 400
    if i in [0, 99, 199, 299, 399]:
        # Plot the rectangle and save the image
        fig, ax = plt.subplots()
        plt.imshow(It1, cmap='gray')
        rect_patch = patches.Rectangle(
            (rect[0], rect[1]), rect[2] - rect[0], rect[3] - rect[1],
            linewidth=2, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect_patch)
        plt.axis('off')
        plt.savefig(f"../results/carseq_{i + 1}.png", bbox_inches='tight', pad_inches=0)

# (num_frames, 4)
np.save("../results/carseqrects.npy", rects)