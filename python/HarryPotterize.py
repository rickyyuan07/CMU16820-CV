import numpy as np
import cv2
import skimage.io 
import skimage.color
from opts import get_opts
from planarH import computeH_ransac, compositeH
from matchPics import matchPics, matchPicsORB, matchPicsSIFT
from displayMatch import displayMatched

# Import necessary functions

# Q2.2.4

def warpImage(opts, template_img, from_img, to_img):
    '''
    Function to warp from_img(hp_cover.jpg) onto to_img(cv_desk.png) using homography
    computed between from template image (cv_cover.jpg) and to_img(cv_desk.png).
    '''

    # Resize from_img such that the warpped image fit the correct size as cv_cover
    from_img = cv2.resize(from_img, (template_img.shape[1], template_img.shape[0]))

    if opts.match_method == 'BRIEF_FAST':
        matches, locs1, locs2 = matchPics(to_img, template_img, opts)
        # Important! Swap x, y axis because cv2.warpPerspective treats x, y differently
        locs1 = locs1[:, ::-1]
        locs2 = locs2[:, ::-1]
    elif opts.match_method == 'ORB':
        matches, locs1, locs2 = matchPicsORB(to_img, template_img, opts)
    elif opts.match_method == 'SIFT':
        matches, locs1, locs2 = matchPicsSIFT(to_img, template_img, opts)
    
    locs1 = locs1[matches[:, 0]]
    locs2 = locs2[matches[:, 1]]
    
    # Homography maps locs2 (template_img) to locs1 (to_img)
    bestH2to1, _ = computeH_ransac(locs1, locs2, opts)
    
    composite_img = compositeH(bestH2to1, from_img, to_img)

    if opts.show_warpped_img:
        cv2.imshow("Composite Image", composite_img)
        cv2.waitKey()

    if opts.store_Q225_img:
        cv2.imwrite(f"../Figures/composite_{opts.max_iters}_{opts.inlier_tol}.png", composite_img)

    return composite_img


if __name__ == "__main__":

    opts = get_opts()

    cv_cover = cv2.imread('../data/cv_cover.jpg')
    cv_desk = cv2.imread('../data/cv_desk.png')
    hp_cover = cv2.imread('../data/hp_cover.jpg')
    warpImage(opts, cv_cover, hp_cover, cv_desk)


