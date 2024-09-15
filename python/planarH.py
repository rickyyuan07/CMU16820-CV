import argparse
from typing import Tuple
import numpy as np
import cv2


def computeH(points1: np.array, points2: np.array) -> np.array:
    '''
    Q2.2.1
    Compute the homography between two sets of points

    Input:
    -----
        points1: numpy array [N, 2], N is the number of correspondences, [x1, y1]
        points2: numpy array [N, 2], N is the number of correspondences, [x2, y2]
    Return
    -----
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

    H2to1 /= H2to1[2, 2]

    return H2to1


def normalize(points: np.array) -> Tuple[np.array, np.array]:
    '''
    Input
    -----
        points: numpy array [N, 2], N is the number of points, (x, y)

    Return
    -----
        normalized points: numpy array [N, 2], N is the number of points, (x, y)
        T: numpy array [3, 3], transformation matrix
    '''
    m = np.mean(points, 0)
    s = np.sqrt(2) / np.linalg.norm(points - m, axis=1).max()
    T = np.array([[s, 0, m[0]],
                  [0, s, m[1]],
                  [0, 0,    1]])
    transformation_mat = np.linalg.inv(T)
    normalized_points = (transformation_mat @ np.concatenate((points, np.ones((points.shape[0], 1))), axis=1).T)
    
    return transformation_mat, normalized_points[:2].T


def computeH_norm(points1: np.array, points2: np.array) -> np.array:
    '''
    Q2.2.2
    Compute the normalized homography matrix H2to1 from points x1 to x2.

    Input
    -----
        points1: numpy array [N, 2], N is the number of correspondences, [x1, y1]
        points2: numpy array [N, 2], N is the number of correspondences, [x2, y2]
    Return
    -----
        H2to1: numpy array [3, 3], normalized homography matrix, H @ [x2, y2, 1]^T = [x1, y1, 1]^T
    '''
    assert points1.shape == points2.shape, "The number of points should be the same"

    T1, points1_norm = normalize(points1)
    T2, points2_norm = normalize(points2)

    H2to1_norm = computeH(points1_norm, points2_norm)
    H2to1 = np.linalg.inv(T1) @ H2to1_norm @ T2

    H2to1 /= H2to1[2, 2]
    return H2to1




def computeH_ransac(locs1: np.array, locs2: np.array, opts: argparse.Namespace) -> Tuple[np.array, np.array]:
    '''
    Q2.2.3
    Compute the best fitting homography mapping points from locs2 to locs1 using RANSAC.

    Input
    -----
        locs1: numpy array [N, 2], N is the number of matching points, [x1, y1]
        locs2: numpy array [N, 2], N is the number of matching points, [x2, y2]
        opts: max_iters: number of iterations to run RANSAC for, 
            inlier_tol: tolerance value for considering a point to be an inlier
    Return
    -----
        bestH2to1: numpy array [3, 3], homography matrix with the most inliers, H @ [x2, y2, 1]^T = [x1, y1, 1]^T
        inliers: numpy array [N], inliers[i] = 1 if the point is an inlier, 0 otherwise
    '''
    max_iters = opts.max_iters
    inlier_tol = opts.inlier_tol

    N = locs1.shape[0]
    bestH2to1 = None
    best_inliers = np.zeros(N)
    max_inliers = 0

    for _ in range(max_iters):
        idx = np.random.choice(N, 4, replace=False)
        H2to1 = computeH_norm(locs1[idx], locs2[idx])

        # Apply homography to all points, calculate the inliners
        locs2_homo = np.hstack((locs2, np.ones((N, 1))))
        mapped_points = (H2to1 @ locs2_homo.T).T
        mapped_points /= mapped_points[:, 2].reshape(-1, 1) # shape: (N, 3)
        
        # Computer errors (L2 norm)
        errors = np.linalg.norm(mapped_points[:, :2] - locs1, axis=1)

        # Count inliers (where error is less than inlier tolerance)
        inliers = (errors < inlier_tol).astype(np.int)
        num_inliers = np.sum(inliers)

        # Update best H
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            bestH2to1 = H2to1
            best_inliers = inliers

    return bestH2to1, best_inliers



def compositeH(H2to1, template, img):
    '''
    Create a composite image after warping the template image on top
    of the image using the homography
    '''
    
    # Create mask of same size as template
    mask = np.ones_like(template, dtype=np.uint8)

    # Warp mask by appropriate homography
    warped_mask = cv2.warpPerspective(mask, H2to1, (img.shape[1], img.shape[0]))
    
    # Warp template by appropriate homography
    warped_template = cv2.warpPerspective(template, H2to1, (img.shape[1], img.shape[0]))

    # Use mask to combine the warped template and the image
    composite_img = np.where(warped_mask > 0, warped_template, img)
    
    return composite_img


