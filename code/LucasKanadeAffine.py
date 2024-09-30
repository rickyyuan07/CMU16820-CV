import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import affine_transform
import cv2

def LucasKanadeAffine(It, It1, threshold, num_iters):
    """
    :param It: template image, shape (H, W) = (240, 320)
    :param It1: Current image, shape (H, W) = (240, 320)
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array] put your implementation here
    """

    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    p = np.zeros(6)
    ################### TODO Implement Lucas Kanade Affine ###################

    h, w = It.shape
    # Create coordinate grid for the entire image
    X, Y = np.meshgrid(np.arange(h), np.arange(w), indexing='ij') # shape: (h, w)

    for _ in range(num_iters):
        ### 1. Warp I with W(x;p) to compute I(W(x;p))
        It1_warp = affine_transform(It1, M, mode='nearest')

        ### 2. Compute the error image
        error = It - It1_warp

        ### 3. Warp the gradient of I with W(x;p) to compute \grad I
        Ix, Iy = np.gradient(It1_warp)

        ### 4. Evaluate the Jacobian dW/dp, Note: we directly compute steepest descent images here
        # J = [[X,0,Y,0,1,0],[0,X,0,Y,0,1]]
        # A = [X*Ix, X*Iy, Y*Ix, Y*Iy, Ix, Iy]
        A = np.vstack((X.flatten()*Ix.flatten(), 
                       X.flatten()*Iy.flatten(), 
                       Y.flatten()*Ix.flatten(), 
                       Y.flatten()*Iy.flatten(), 
                       Ix.flatten(), 
                       Iy.flatten())).T # shape: (w*h, 6)

        ### 5. Compute the approximated Hessian H = \sum_{x} \grad I \grad I^T (A^T @ A)
        H = A.T @ A # shape: (6, 6)

        ### 6. Computer delta p (dp) = H^{-1} @ A^T @ error
        dp = np.linalg.inv(H) @ A.T @ error.flatten() # dp matrix, shape: (6,)
        
        ### 7. Update warp parameters p <- p + dp
        p += dp
        M = np.array([[1.0 + p[0], p[2], p[4]], [p[1], 1.0 + p[3], p[5]]])

        # Terminate if the change in p (dp) is below the threshold
        if np.linalg.norm(dp) < threshold:
            # print("iter", _, "|dp|=", np.linalg.norm(dp), "sum(err)=", np.sum(error)) # DEBUG
            break
    
    return M
