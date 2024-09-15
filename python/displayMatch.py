import numpy as np
import cv2
from matchPics import matchPics, matchPicsORB, matchPicsSIFT
from helper import plotMatches
from opts import get_opts


def displayMatched(opts, image1, image2):
    """
    Displays matches between two images

    Input
    -----
    opts: Command line args
    image1, image2: Source images
    """

    if opts.match_method == 'BRIEF_FAST':
        matches, locs1, locs2 = matchPics(image1, image2, opts)
    elif opts.match_method == 'ORB':
        matches, locs1, locs2 = matchPicsORB(image1, image2, opts)
        locs1 = locs1[:, ::-1]
        locs2 = locs2[:, ::-1]
    elif opts.match_method == 'SIFT':
        matches, locs1, locs2 = matchPicsSIFT(image1, image2, opts)
        locs1 = locs1[:, ::-1]
        locs2 = locs2[:, ::-1]

    #display matched features
    plotMatches(image1, image2, matches, locs1, locs2)

if __name__ == "__main__":

    opts = get_opts()
    image1 = cv2.imread('../data/cv_cover.jpg')
    image2 = cv2.imread('../data/cv_desk.png')

    displayMatched(opts, image1, image2)
