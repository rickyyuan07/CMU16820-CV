import numpy as np
import matplotlib.pyplot as plt

from helper import displayEpipolarF, calc_epi_error, toHomogenous
from q2_1_eightpoint import eightpoint
from q2_2_sevenpoint import sevenpoint
from q3_2_triangulate import findM2

import scipy

# Insert your package here
from scipy.optimize import minimize
from tqdm import tqdm

# Helper functions for this assignment. DO NOT MODIFY!!!
"""
Helper functions.

Written by Chen Kong, 2018.
Modified by Zhengyi (Zen) Luo, 2021
"""


def plot_3D_dual(P_before, P_after):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Blue: before; red: after")
    ax.scatter(P_before[:, 0], P_before[:, 1], P_before[:, 2], c="blue")
    ax.scatter(P_after[:, 0], P_after[:, 1], P_after[:, 2], c="red")
    while True:
        x, y = plt.ginput(1, mouse_stop=2)[0]
        plt.draw()


"""
Q5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
            nIters, Number of iterations of the Ransac
            tol, tolerence for inliers
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers

    Hints:
    (1) You can use the calc_epi_error from q1 with threshold to calcualte inliers. Tune the threshold based on 
        the results/expected number of inliners. You can also define your own metric. 
    (2) Use the seven point alogrithm to estimate the fundamental matrix as done in q1
    (3) Choose the resulting F that has the most number of inliers
    (4) You can increase the nIters to bigger/smaller values
 
"""


def ransacF(pts1, pts2, M, nIters=1000, tol=10):
    MODE = "EIGHT" # Change to "SEVEN" for seven point algorithm
    N = pts1.shape[0] # Number of points

    pts1_homogenous, pts2_homogenous = toHomogenous(pts1), toHomogenous(pts2)
    bestF = np.zeros((3, 3))
    best_inliers = np.zeros(N, dtype=bool)

    for _ in tqdm(range(nIters)):
        if MODE == "EIGHT":
            idx = np.random.choice(N, 8) # Should be replace=False, but somehow it doesn't work
            F = eightpoint(pts1[idx], pts2[idx], M)
        else: # use seven point algorithm
            idx = np.random.choice(N, 7)
            F = sevenpoint(pts1[idx], pts2[idx], M)
        
        inliers = calc_epi_error(pts1_homogenous, pts2_homogenous, F) < tol

        if np.sum(inliers) > np.sum(best_inliers):
            best_inliers = inliers
            bestF = F

    bestF = eightpoint(pts1[best_inliers], pts2[best_inliers], M)
    return bestF, best_inliers
"""
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
"""


# https://krasjet.github.io/quaternion/quaternion.pdf
# https://courses.cs.duke.edu//fall13/compsci527/notes/rodrigues.pdf
def rodrigues(r):
    theta = np.linalg.norm(r) # Rotation angle is the magnitude of r

    if theta < 1e-10: # Close to zero
        return np.identity(3)

    u = r / theta
    u_cross = np.array([
        [0, -u[2], u[1]],
        [u[2], 0, -u[0]],
        [-u[1], u[0], 0]
    ])

    R = np.cos(theta) * np.identity(3) + (1 - np.cos(theta)) * np.outer(u, u) + np.sin(theta) * u_cross
    return R


"""
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
"""


def invRodrigues(R):
    # Make sure R is a valid rotation matrix
    assert np.allclose(R.T @ R, np.identity(3), atol=1e-6), "R is not a valid rotation matrix, R^T * R != I"
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-6), "R is not a valid rotation matrix, det(R) != 1"

    A = (R - R.T) / 2
    rho = np.array([A[2, 1], A[0, 2], A[1, 0]]).T
    s = np.linalg.norm(rho)
    c = (np.trace(R) - 1) / 2

    if np.isclose(s, 0) and np.isclose(c, 1):
        return np.zeros(3)
    if np.isclose(s, 0) and np.isclose(c, -1):
        # Let v = a nonzero column of R + I
        M = (R + np.eye(3)) / 2
        for i in range(3):
            if np.linalg.norm(M[:, i]) > 1e-10:
                v = M[:, i]
                break
        
        u = v / np.linalg.norm(v)
        r = np.pi * u
        # Construct S_{1/2}(u * pi)
        if np.isclose(np.linalg.norm(r), np.pi) and ((r[0] == r[1] == 0 and r[2] < 0)
            or (r[0] == 0 and r[1] < 0) or r[0] < 0):
            r = -r
    
    u = rho / s
    theta = np.arctan2(s, c)
    r = theta * u
    return r

"""
Q5.3: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
"""


def rodriguesResidual(K1, M1, p1, K2, p2, x):
    N = p1.shape[0]
    P = x[:3 * N].reshape(N, 3)  # 3D points, shape (N, 3)
    r2 = x[3 * N:3 * N + 3]      # Rotation vector, shape (3,)
    t2 = x[3 * N + 3:]           # Translation vector, shape (3,)

    R2 = rodrigues(r2)
    # M2 = [R2 | t2]
    M2 = np.hstack((R2, t2.reshape(-1, 1)))

    C1 = K1 @ M1 # Camera 1 projection matrix
    C2 = K2 @ M2 # Camera 2 projection matrix

    P_homo = np.hstack((P, np.ones((N, 1)))) # Convert 3D points to homogeneous coordinates, shape (N, 4)
    p1_hat_homo = (C1 @ P_homo.T).T # 2D point projected to image 1, shape (N, 3)
    p2_hat_homo = (C2 @ P_homo.T).T # 2D point projected to image 2, shape (N, 3)

    p1_hat = p1_hat_homo[:, :2] / p1_hat_homo[:, 2:3] # shape (N, 2)
    p2_hat = p2_hat_homo[:, :2] / p2_hat_homo[:, 2:3] # shape (N, 2)

    # Compute residuals as the difference between original image projections and estimated projections
    residuals = np.concatenate([(p1 - p1_hat).reshape(-1), (p2 - p2_hat).reshape(-1)])
    return residuals


"""
Q5.3 Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
            o1, the starting objective function value with the initial input
            o2, the ending objective function value after bundle adjustment

    Hints:
    (1) Use the scipy.optimize.minimize function to minimize the objective function, rodriguesResidual. 
        You can try different (method='..') in scipy.optimize.minimize for best results. 
"""


def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    N = p1.shape[0] # Number of points

    # Extract the rotation and translation from the initial extrinsics of camera 2
    R2_init = M2_init[:, :3]
    t2_init = M2_init[:, 3]
    # Convert initial rotation matrix to Rodrigues vector
    r2_init = invRodrigues(R2_init)

    # Concatenate 3D points, rotation vector, and translation for rodriguesResidual
    x0 = np.concatenate([P_init.flatten(), r2_init, t2_init]) # (3N + 3 + 3,)

    # Use sum of squared residuals as the objective function
    def objective(x):
        residuals = rodriguesResidual(K1, M1, p1, K2, p2, x)
        return np.sum(residuals**2)

    # Run optimization using Scipy's minimize, use default BFGS algorithm
    result = minimize(objective, x0, method='BFGS', options={'maxiter': 1000, 'disp': False})

    # Extract optimized parameters
    x_opt = result.x
    obj_start = objective(x0) # Initial objective function value
    obj_end = result.fun      # Optimized objective function value

    # Extract optimized 3D points, rotation vector, and translation
    P_opt = x_opt[:3 * N].reshape(N, 3)
    r2_opt = x_opt[3 * N:3 * N + 3]
    t2_opt = x_opt[3 * N + 3:]

    R2_opt = rodrigues(r2_opt) # Back to rotation matrix
    M2 = np.hstack((R2_opt, t2_opt.reshape(-1, 1)))

    return M2, P_opt, obj_start, obj_end


if __name__ == "__main__":
    np.random.seed(1)  # Added for testing, can be commented out

    some_corresp_noisy = np.load(
        "data/some_corresp_noisy.npz"
    )  # Loading correspondences
    intrinsics = np.load("data/intrinsics.npz")  # Loading the intrinscis of the camera
    K1, K2 = intrinsics["K1"], intrinsics["K2"]
    noisy_pts1, noisy_pts2 = some_corresp_noisy["pts1"], some_corresp_noisy["pts2"]
    im1 = plt.imread("data/im1.png")
    im2 = plt.imread("data/im2.png")

    F, inliers = ransacF(noisy_pts1, noisy_pts2, M=np.max([*im1.shape, *im2.shape]))

    # displayEpipolarF(im1, im2, F)

    # Simple Tests to verify your implementation:
    pts1_homogenous, pts2_homogenous = toHomogenous(noisy_pts1), toHomogenous(
        noisy_pts2
    )

    assert F.shape == (3, 3)
    assert F[2, 2] == 1
    assert np.linalg.matrix_rank(F) == 2

    # Simple Tests to verify your implementation:
    from scipy.spatial.transform import Rotation as sRot

    rotVec = sRot.random()
    mat = rodrigues(rotVec.as_rotvec())

    assert np.linalg.norm(rotVec.as_rotvec() - invRodrigues(mat)) < 1e-3
    assert np.linalg.norm(rotVec.as_matrix() - mat) < 1e-3

    # Visualization:
    np.random.seed(1)
    correspondence = np.load(
        "data/some_corresp_noisy.npz"
    )  # Loading noisy correspondences
    intrinsics = np.load("data/intrinsics.npz")  # Loading the intrinscis of the camera
    K1, K2 = intrinsics["K1"], intrinsics["K2"]
    pts1, pts2 = correspondence["pts1"], correspondence["pts2"]
    im1 = plt.imread("data/im1.png")
    im2 = plt.imread("data/im2.png")
    M = np.max([*im1.shape, *im2.shape])

    # TODO: YOUR CODE HERE
    """
    Call the ransacF function to find the fundamental matrix
    Call the findM2 function to find the extrinsics of the second camera
    Call the bundleAdjustment function to optimize the extrinsics and 3D points
    Plot the 3D points before and after bundle adjustment using the plot_3D_dual function
    """

    ## Compare the result of RANSAC with the result of the 8 point run on the noisy data
    F, inliers = ransacF(pts1, pts2, M, nIters=200, tol=3)
    error = np.sum(calc_epi_error(pts1_homogenous[inliers], pts2_homogenous[inliers], F))
    print(f"Error for RANSAC: {error}")

    F_vanilla = eightpoint(pts1, pts2, M)
    error_vanilla = np.sum(calc_epi_error(pts1_homogenous[inliers], pts2_homogenous[inliers], F_vanilla))
    print(f"Error for 8 point: {error_vanilla}")
    print(f"Inliers: {np.sum(inliers) / len(inliers)}")

    # Run findM2 based on RANSAC result and inliers
    M1 = np.hstack((np.identity(3), np.zeros((3,1))))
    M2, C2, P = findM2(F, pts1[inliers], pts2[inliers], intrinsics, None)

    # Run bundle adjustment, compare the reprojection error before and after optimization
    M2_ba, P_ba, obj_start, obj_end = bundleAdjustment(K1, M1, pts1[inliers], K2, M2, pts2[inliers], P)
    print(f"Reprojection error before optimization: {obj_start :.6f}\nReprojection error after optimization: {obj_end :.6f}")

    plot_3D_dual(P, P_ba)