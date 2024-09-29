import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from SubtractDominantMotion import SubtractDominantMotion

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument(
    '--num_iters', type=int, default=1e3, help='number of iterations of Lucas-Kanade'
)
parser.add_argument(
    '--threshold',
    type=float,
    default=1e-2,
    help='dp threshold of Lucas-Kanade for terminating optimization',
)
parser.add_argument(
    '--tolerance',
    type=float,
    default=0.2,
    help='binary threshold of intensity difference when computing the mask',
)
parser.add_argument(
    '--seq_file',
    default='../data/antseq.npy',
)

# Best: Threshold=0.001; Tolerance=0.01; Erosion/Dilation Iteration=1
args = parser.parse_args()
num_iters = int(args.num_iters)
threshold = args.threshold
tolerance = args.tolerance
seq_file = args.seq_file

seq = np.load(seq_file)

'''
HINT:
1. Create an empty array 'masks' to store the motion masks for each frame.
2. Set the initial mask for the first frame to False.
3. Use the SubtractDominantMotion function to compute the motion mask between consecutive frames.
4. Use the motion 'masks; array for visualization.
'''

masks = np.zeros_like(seq, dtype=bool)
frames_to_show = [30, 60, 90, 120]
# Compute motion masks
for i in frames_to_show:
    masks[:,:,i] = SubtractDominantMotion(seq[:,:,i-1], seq[:,:,i], threshold, num_iters, tolerance)

# Visualize and save the results
fig, axes = plt.subplots(1, 4, figsize=(15, 5))
for i, frame in enumerate(frames_to_show):
    ax = axes[i]
    
    # Create a RGB image: original grayscale image in all channels
    rgb_image = np.stack([seq[:,:,frame]]*3, axis=-1)
    
    # Overlay the mask in blue
    rgb_image[masks[:,:,frame], 0] = 0.0  # Red channel
    rgb_image[masks[:,:,frame], 1] = 0.0  # Green channel
    rgb_image[masks[:,:,frame], 2] = 1.0  # Blue channel
    
    ax.imshow(rgb_image)
    ax.set_title(f'Frame {frame}')
    ax.axis('off')

plt.tight_layout()
plt.savefig('./result/q2_3_ant.png')
plt.show()

# DEBUG
for frame in frames_to_show:
    print(f"Frame {frame}: {masks[:,:,frame].sum()} pixels detected as moving")