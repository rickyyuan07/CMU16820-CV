import numpy as np
import matplotlib.pyplot as plt

from helper import camera2
from q2_1_eightpoint import eightpoint
from q3_1_essential_matrix import essentialMatrix

# Insert your package here


"""
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.

    Hints:
    (1) For every input point, form A using the corresponding points from pts1 & pts2 and C1 & C2
    (2) Solve for the least square solution using np.linalg.svd
    (3) Calculate the reprojection error using the calculated 3D points and C1 & C2 (do not forget to convert from 
        homogeneous coordinates to non-homogeneous ones)
    (4) Keep track of the 3D points and projection error, and continue to next point 
    (5) You do not need to follow the exact procedure above. 
"""


def triangulate(C1, pts1, C2, pts2):
    x1, y1 = pts1[:, 0], pts1[:, 1]
    x2, y2 = pts2[:, 0], pts2[:, 1]
    N = x1.shape[0] # Number of points

    # Construct skew-symmetric matrices for all points
    skew_symmetric1 = np.array([
        [np.zeros(N), -1.0 * np.ones(N), y1],
        [np.ones(N), np.zeros(N), -1.0 * x1],
        [-1.0 * y1, x1, np.zeros(N)]
    ]).transpose(2, 0, 1) # Shape (N, 3, 3)

    skew_symmetric2 = np.array([
        [np.zeros(N), -1.0 * np.ones(N), y2],
        [np.ones(N), np.zeros(N), -1.0 * x2],
        [-1.0 * y2, x2, np.zeros(N)]
    ]).transpose(2, 0, 1) # Shape (N, 3, 3)

    # Only consider the first two rows because the third row is linearly dependent
    eq1 = (skew_symmetric1 @ C1)[:, :2, :]  # Shape (N, 2, 4)
    eq2 = (skew_symmetric2 @ C2)[:, :2, :]  # Shape (N, 2, 4)

    A = np.concatenate([eq1, eq2], axis=1)  # Shape (N, 4, 4)

    # Initialize the 3D points matrix (in homogeneous coordinates)
    P = np.zeros((N, 4))
    for i in range(N):
        _, _, V = np.linalg.svd(A[i]) # V has shape (4, 4)
        omega = V[-1]
        P[i, :] = omega / omega[-1] # Normalize the homogeneous coordinates

    # Project the 3D points back to 2D
    proj1 = C1 @ P.T  # Shape (3, N)
    proj2 = C2 @ P.T  # Shape (3, N)

    # Normalize, back to non-homogeneous coordinates
    proj1 = proj1[:2, :] / proj1[2, :]  # Shape (2, N)
    proj2 = proj2[:2, :] / proj2[2, :]  # Shape (2, N)

    # Compute the reprojection error
    err = np.sum(np.linalg.norm(proj1.T - pts1, axis=1) + np.linalg.norm(proj2.T - pts2, axis=1))

    return P[:, :3], err


"""
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
"""


def findM2(F, pts1, pts2, intrinsics, filename="q3_3.npz"):
    """
    Q2.2: Function to find camera2's projective matrix given correspondences
        Input:  F, the pre-computed fundamental matrix
                pts1, the Nx2 matrix with the 2D image coordinates per row
                pts2, the Nx2 matrix with the 2D image coordinates per row
                intrinsics, the intrinsics of the cameras, load from the .npz file
                filename, the filename to store results
        Output: [M2, C2, P] the computed M2 (3x4) camera projective matrix, C2 (3x4) K2 * M2, and the 3D points P (Nx3)

    ***
    Hints:
    (1) Loop through the 'M2s' and use triangulate to calculate the 3D points and projection error. Keep track
        of the projection error through best_error and retain the best one.
    (2) Remember to take a look at camera2 to see how to correctly reterive the M2 matrix from 'M2s'.

    """
    K1, K2 = intrinsics["K1"], intrinsics["K2"]
    E = essentialMatrix(F, K1, K2)

    # Get the 4 candidates of M2s
    M2s = camera2(E)

    # C1 = K1 @ M1 = K1 @ [I|0]
    M1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    C1 = K1 @ M1

    best_M2, best_C2, best_P = None, None, None
    most_pos = 0
    for i in range(4):
        # C2 = K2 @ M2 = K2 @ [R|t]
        M2 = M2s[:,:,i]
        C2 = K2 @ M2
        P, _ = triangulate(C1, pts1, C2, pts2)

        # Check if the 3D points are in front of the camera
        if np.sum(P[:, -1] > 0) > most_pos:
            best_M2, best_C2, best_P = M2, C2, P
            most_pos = np.sum(P[:, -1] > 0)
            break

    if filename is not None:
        np.savez(filename, M2=best_M2, C2=best_C2, P=best_P)

    return best_M2, best_C2, best_P


if __name__ == "__main__":
    correspondence = np.load("data/some_corresp.npz")  # Loading correspondences
    intrinsics = np.load("data/intrinsics.npz")  # Loading the intrinscis of the camera
    K1, K2 = intrinsics["K1"], intrinsics["K2"]
    pts1, pts2 = correspondence["pts1"], correspondence["pts2"]
    im1 = plt.imread("data/im1.png")
    im2 = plt.imread("data/im2.png")

    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))

    M2, C2, P = findM2(F, pts1, pts2, intrinsics)

    # Simple Tests to verify your implementation:
    M1 = np.hstack((np.identity(3), np.zeros(3)[:, np.newaxis]))
    C1 = K1.dot(M1)
    C2 = K2.dot(M2)
    P_test, err = triangulate(C1, pts1, C2, pts2)
    print(err)
    assert err < 500
