import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from matchPics import matchPics
from opts import get_opts
from tqdm import tqdm

#Q2.1.6

def rotTest(opts):
    # Read the image and convert to grayscale
    image = cv2.imread('../data/cv_cover.jpg')
    
    # Initialize histogram
    rotations = np.arange(0, 360, 10)
    match_counts = np.zeros(36)

    for angle in tqdm(rotations):
        # Rotate Image, reshape=False because it's easier to match features(?)
        rotated_image = rotate(image, angle=angle, reshape=False)

        # Compute features, descriptors and Match features
        matches, locs1, locs2 = matchPics(image, rotated_image, opts)

        # Update histogram
        # (Assuming you have a histogram variable to update)
        # histogram[i] = len(matches)
        match_counts[angle//10] = len(matches)

    # Display histogram
    plt.figure()
    plt.bar(rotations, match_counts, width=8)
    plt.xlabel('Rotation (degrees)')
    plt.ylabel('Number of Matches')
    plt.title('BRIEF Descriptor: Number of Matches vs. Rotation')
    plt.show()


if __name__ == "__main__":

    opts = get_opts()
    rotTest(opts)
