import numpy as np
import cv2
import skimage.color
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection

# Q2.1.4

def matchPics(I1, I2, opts):
    """
    Match features across images

    Input
    -----
    I1, I2: Source images
    opts: Command line args

    Returns
    -------
    matches: List of indices of matched features across I1, I2 [p x 2]
    locs1, locs2: Pixel coordinates of matches [N x 2]
    """

    ratio = opts.ratio  # ratio for BRIEF feature descriptor
    sigma = opts.sigma  # threshold for corner detection using FAST feature detector

    # Convert images to grayscale
    I1_gray = skimage.color.rgb2gray(I1)
    I2_gray = skimage.color.rgb2gray(I2)

    # Detect features in both images using the FAST detector
    locs1 = corner_detection(I1_gray, sigma)
    locs2 = corner_detection(I2_gray, sigma)

    # Obtain BRIEF descriptors for the computed feature locations
    desc1, locs1 = computeBrief(I1_gray, locs1)
    desc2, locs2 = computeBrief(I2_gray, locs2)

    # Match features using the BRIEF descriptors
    matches = briefMatch(desc1, desc2, ratio)
    return matches, locs1, locs2


# For better point representation at Q3 AR
def matchPicsORB(I1, I2, opts):
    """
    Match features across images using ORB

    Input
    -----
    I1, I2: Source images
    opts: Command line args

    Returns
    -------
    matches: List of indices of matched features across I1, I2 [p x 2]
    locs1, locs2: Pixel coordinates of matches [N x 2]
    """
    orb = cv2.ORB_create(nfeatures=5000)
    kp1, des1 = orb.detectAndCompute(I1, None)
    kp2, des2 = orb.detectAndCompute(I2, None)

    # create BFMatcher object (Brute Force Matcher with Hamming distance)
    # Note: crossCheck=True means that the BFMatcher returns the best match in both images
    # crossCheck=False to save computation time
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply Lowe's ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Get the pixel coordinates of the matches
    locs1 = np.array([kp1[m.queryIdx].pt for m in good_matches])
    locs2 = np.array([kp2[m.trainIdx].pt for m in good_matches])

    mm = np.array([[i, i] for i in range(len(locs1))])
    return mm, locs1, locs2


def matchPicsSIFT(I1, I2, opts):
    """
    Match features across images using SIFT

    Input
    -----
    I1, I2: Source images
    opts: Command line args

    Returns
    -------
    matches: List of indices of matched features across I1, I2 [p x 2]
    locs1, locs2: Pixel coordinates of matches [N x 2]
    """
    sift = cv2.SIFT_create()             # opencv-python==4.5.1.48
    kp1, des1 = sift.detectAndCompute(I1, None)
    kp2, des2 = sift.detectAndCompute(I2, None)

    # create BFMatcher object (Brute Force Matcher with Hamming distance)
    # Note: crossCheck=True means that the BFMatcher returns the best match in both images
    # crossCheck=False to save computation time
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply Lowe's ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Get the pixel coordinates of the matches
    locs1 = np.array([kp1[m.queryIdx].pt for m in good_matches])
    locs2 = np.array([kp2[m.trainIdx].pt for m in good_matches])

    mm = np.array([[i, i] for i in range(len(locs1))])
    return mm, locs1, locs2