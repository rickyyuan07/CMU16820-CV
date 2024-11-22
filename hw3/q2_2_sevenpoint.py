import numpy as np
import matplotlib.pyplot as plt

from helper import displayEpipolarF, calc_epi_error, toHomogenous, _singularize, refineF

# Insert your package here
from tqdm import tqdm


"""
Q2.2: Seven Point Algorithm for calculating the fundamental matrix
    Input:  pts1, 7x2 Matrix containing the corresponding points from image1
            pts2, 7x2 Matrix containing the corresponding points from image2
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated 3x3 fundamental matrixes.
    
    HINTS:
    (1) Normalize the input pts1 and pts2 scale paramter M.
    (2) Setup the seven point algorithm's equation.
    (3) Solve for the least square solution using SVD. 
    (4) Pick the last two colum vector of vT.T (the two null space solution f1 and f2)
    (5) Use the singularity constraint to solve for the cubic polynomial equation of  F = a*f1 + (1-a)*f2 that leads to 
        det(F) = 0. Solving this polynomial will give you one or three real solutions of the fundamental matrix. 
        Use np.polynomial.polynomial.polyroots to solve for the roots
    (6) Unscale the fundamental matrixes and return as Farray
"""


def sevenpoint(pts1, pts2, M):
    T = np.array([[1 / M, 0, 0], [0, 1 / M, 0], [0, 0, 1]])
    pts1, pts2 = pts1 / M, pts2 / M # Normalizing the points
    x1, y1 = pts1[:, 0], pts1[:, 1]
    x2, y2 = pts2[:, 0], pts2[:, 1]

    A = np.array([x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, np.ones_like(x1)]).T # (N, 9)

    _, _, V = np.linalg.svd(A)
    F1 = V[-1].reshape(3, 3)
    F2 = V[-2].reshape(3, 3)
    
    # Construct the polynomial for solving alpha
    f = lambda alpha: np.linalg.det(alpha * F1 + (1 - alpha) * F2)
    coeff = [0] * 4
    coeff[0] = (f(2) - f(-2) - 2 * (f(1) - f(-1))) / 12
    coeff[1] = (f(1) + f(-1)) / 2 - f(0)
    coeff[2] = (f(1) - f(-1) - 2 * coeff[0]) / 2
    coeff[3] = f(0)
    
    # Might have 1 or 3 real roots
    roots = np.roots(coeff)
    roots = [root for root in roots if np.isreal(root)]

    # Construct back to F
    Farray = [a * F1 + (1 - a) * F2 for a in roots]
    # Singularize and refine
    Farray = [_singularize(F) for F in Farray]
    Farray = [refineF(F, pts1, pts2) for F in Farray]
    # Unscale, and force F[2, 2] = 1
    Farray = [T.T @ F @ T for F in Farray]
    Farray = [F / F[2, 2] for F in Farray]

    return Farray


if __name__ == "__main__":
    correspondence = np.load("data/some_corresp.npz")  # Loading correspondences
    intrinsics = np.load("data/intrinsics.npz")  # Loading the intrinscis of the camera
    K1, K2 = intrinsics["K1"], intrinsics["K2"]
    pts1, pts2 = correspondence["pts1"], correspondence["pts2"]
    im1 = plt.imread("data/im1.png")
    im2 = plt.imread("data/im2.png")

    # indices = np.arange(pts1.shape[0])
    # indices = np.random.choice(indices, 7, False)
    indices = np.array([82, 19, 56, 84, 54, 24, 18])

    M = np.max([*im1.shape, *im2.shape])

    Farray = sevenpoint(pts1[indices, :], pts2[indices, :], M)

    print(Farray)

    F = Farray[0]

    np.savez("q2_2.npz", F, M)

    # fundamental matrix must have rank 2!
    # assert(np.linalg.matrix_rank(F) == 2)

    # Simple Tests to verify your implementation:
    # Test out the seven-point algorithm by randomly sampling 7 points and finding the best solution.
    np.random.seed(1)  # Added for testing, can be commented out

    pts1_homogenous, pts2_homogenous = toHomogenous(pts1), toHomogenous(pts2)

    max_iter = 500
    pts1_homo = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
    pts2_homo = np.hstack((pts2, np.ones((pts2.shape[0], 1))))

    ress = []
    F_res = []
    choices = []
    M = np.max([*im1.shape, *im2.shape])
    for i in tqdm(range(max_iter)):
        choice = np.random.choice(range(pts1.shape[0]), 7)
        pts1_choice = pts1[choice, :]
        pts2_choice = pts2[choice, :]
        Fs = sevenpoint(pts1_choice, pts2_choice, M)
        for F in Fs:
            choices.append(choice)
            res = calc_epi_error(pts1_homo, pts2_homo, F)
            F_res.append(F)
            ress.append(np.mean(res))

    min_idx = np.argmin(np.abs(np.array(ress)))
    F = F_res[min_idx]
    print("Error:", ress[min_idx])
    print(F)

    displayEpipolarF(im1, im2, F)
    assert F.shape == (3, 3)
    assert F[2, 2] == 1
    assert np.linalg.matrix_rank(F) == 2
    assert np.mean(calc_epi_error(pts1_homogenous, pts2_homogenous, F)) < 1
