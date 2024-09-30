import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import affine_transform
import cv2

def InverseCompositionAffine(It, It1, threshold, num_iters):
    """
    :param It: template image, shape = (H, W)
    :param It1: Current image, shape = (H, W)
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array]
    """

    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    p = np.zeros(6)
    ################### TODO Implement Lucas Kanade Affine ###################
    h, w = It.shape
    # Create coordinate grid for the entire image
    X, Y = np.meshgrid(np.arange(h), np.arange(w), indexing='ij') # shape: (h, w)

    ### 1. Evaluate the gradient of the template image
    Ix, Iy = np.gradient(It)

    ### 2. Evaluate the Jacobian dW/dp at (x;0); 3. Compute the steepest descent images
    A = np.vstack((X.flatten()*Ix.flatten(),
                     X.flatten()*Iy.flatten(),
                     Y.flatten()*Ix.flatten(),
                     Y.flatten()*Iy.flatten(),
                     Ix.flatten(),
                     Iy.flatten())).T # shape: (w*h, 6)
    
    ### 4. Compute the inverse of the Hessian matrix
    H_inv = np.linalg.inv(A.T @ A)

    for _ in range(num_iters):
        ### 1. Warp I with W(x;p) to compute I(W(x;p))
        It1_warp = affine_transform(It1, M, mode='nearest')

        ### 2. Compute the error image
        error = It1_warp - It

        ### 3. Warp the gradient of I with W(x;p) to compute \grad I
        Ix, Iy = np.gradient(It1_warp)

        ### 4. Computer delta p (dp) = H^{-1} @ A^T @ error
        dp = H_inv @ A.T @ error.flatten()
        
        ### 5. Update warp parameters W(x;p) = W(x;p) @ W(x;dp)
        dM = np.array([[1.0 + dp[0], dp[2], dp[4]], [dp[1], 1.0 + dp[3], dp[5]], [0.0, 0.0, 1.0]])
        M = M @ np.linalg.inv(dM)

        # Terminate if the change in p (dp) is below the threshold
        if np.linalg.norm(dp) < threshold:
            # print("iter", _, "|dp|=", np.linalg.norm(dp), "sum(err)=", np.sum(error)) # DEBUG
            break
    
    return M
