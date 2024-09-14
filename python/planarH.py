import numpy as np
import cv2


def computeH(points1: np.array, points2: np.array) -> np.array:
    '''
    #Q2.2.1
    # TODO: Compute the homography between two sets of points
    Input:
        points1: numpy array [N, 2], N is the number of correspondences, [x1, y1]
        points2: numpy array [N, 2], N is the number of correspondences, [x2, y2]
    Return:
        H2to1: numpy array [3, 3], homography matrix, H @ [x2, y2, 1]^T = [x1, y1, 1]^T
    '''
    assert points1.shape == points2.shape, "The number of points should be the same"
    n_points = points1.shape[0]
    A = np.zeros((2*n_points, 9), dtype=np.float64)

    for i in range(n_points):
        x1, y1 = points1[i]
        x2, y2 = points2[i]
        A[2*i] = [-x2, -y2, -1, 0, 0, 0, x1*x2, x1*y2, x1]
        A[2*i+1] = [0, 0, 0, -x2, -y2, -1, y1*x2, y1*y2, y1]

    U, S, V = np.linalg.svd(A)
    H2to1 = V[-1].reshape(3, 3)

    # Note that the homography matrix is not normalized
    # H = H / H[-1, -1] # normalize
    return H2to1


def computeH_norm(x1, x2):
    #Q2.2.2
    # TODO: Compute the centroid of the points


    # TODO: Shift the origin of the points to the centroid


    # TODO: Normalize the points so that the largest distance from the origin is equal to sqrt(2)


    # TODO: Similarity transform 1


    # TODO: Similarity transform 2


    # TODO: Compute homography


    # TODO: Denormalization
    

    return H2to1




def computeH_ransac(locs1, locs2, opts):
    #Q2.2.3
    #Compute the best fitting homography given a list of matching points
    max_iters = opts.max_iters  # the number of iterations to run RANSAC for
    inlier_tol = opts.inlier_tol # the tolerance value for considering a point to be an inlier

    


    return bestH2to1, inliers



def compositeH(H2to1, template, img):
    
    #Create a composite image after warping the template image on top
    #of the image using the homography

    #Note that the homography we compute is from the image to the template;
    #x_template = H2to1*x_photo
    #For warping the template to the image, we need to invert it.
    

    # TODO: Create mask of same size as template

    # TODO: Warp mask by appropriate homography

    # TODO: Warp template by appropriate homography

    # TODO: Use mask to combine the warped template and the image
    
    return composite_img


