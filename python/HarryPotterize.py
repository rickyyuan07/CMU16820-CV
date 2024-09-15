import numpy as np
import cv2
import skimage.io 
import skimage.color
from opts import get_opts
from planarH import computeH_ransac, compositeH
from matchPics import matchPics
from displayMatch import displayMatched

# Import necessary functions

# Q2.2.4

def warpImage(opts):
    '''
    Function to warp 'hp_cover.jpg' onto 'cv_desk.png' using homography
    computed between from 'cv_cover.jpg' and 'cv_desk.png'.
    '''
    cv_cover = cv2.imread('../data/cv_cover.jpg')
    cv_desk = cv2.imread('../data/cv_desk.png')
    hp_cover = cv2.imread('../data/hp_cover.jpg')

    # Resize hp_cover such that the warpped image fit the correct size as cv_cover
    hp_cover = cv2.resize(hp_cover, (cv_cover.shape[1], cv_cover.shape[0]))

    matches, locs1, locs2 = matchPics(cv_desk, cv_cover, opts)
    locs1 = locs1[matches[:, 0]]
    locs2 = locs2[matches[:, 1]]
    # Important! Swap x, y axis because cv2.warpPerspective treats x, y differently
    locs1 = locs1[:, ::-1]
    locs2 = locs2[:, ::-1]
    
    # Homography maps locs2 (cv_cover) to locs1 (cv_desk)
    bestH2to1, _ = computeH_ransac(locs1, locs2, opts)
    
    composite_img = compositeH(bestH2to1, hp_cover, cv_desk)
    
    cv2.imshow("Composite Image", composite_img)
    cv2.waitKey()
    cv2.imwrite("../Figures/composite.png", composite_img)



if __name__ == "__main__":

    opts = get_opts()
    warpImage(opts)


