import numpy as np
from scipy.interpolate import RectBivariateSpline
import cv2
import matplotlib.pyplot as plt

def LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros(2)):
    """
    :param It: template image, shape (H, W) = (240, 320)
    :param It1: Current image, shape (H, W) = (240, 320)
    :param rect: Current position of the car (top left, bot right coordinates), inclusive
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :param p0: Initial movement vector [dp_x0, dp_y0]
    :return: p: movement vector [dp_x, dp_y]
    """
	
    # Put your implementation here
    # set up the threshold
    ################### TODO Implement Lucas Kanade ###################
    # Initialize the warp parameters p
    p = p0.copy()
    x1, y1, x2, y2 = rect

    # Note: change to col-major order
    It = It.T
    It1 = It1.T

    # Generate spline for both It (template image) and It1 (current image)
    spline_It = RectBivariateSpline(np.arange(It.shape[0]), np.arange(It.shape[1]), It)
    spline_It1 = RectBivariateSpline(np.arange(It1.shape[0]), np.arange(It1.shape[1]), It1)

    # Create meshgrid for the rectangle region
    x = np.arange(x1, x2 + 1)
    y = np.arange(y1, y2 + 1)
    X, Y = np.meshgrid(x, y) # shape: (36, 87), (36, 87)

    # Extract the template image region using the spline
    template = spline_It.ev(X, Y) # shape: (36, 87)

    for _ in range(num_iters):
        ### 1. Warp I with W(x;p) to compute I(W(x;p))
        X_warp, Y_warp = X + p[0], Y + p[1]
        # Ensure that the warped coordinates are within bounds of the current image
        X_warp = np.clip(X_warp, 0, It1.shape[0] - 1)
        Y_warp = np.clip(Y_warp, 0, It1.shape[1] - 1)

        It1_warp = spline_It1.ev(X_warp, Y_warp) # shape: (36, 87)

        ### 2. Compute the error image
        error = template - It1_warp # shape: (36, 87)

        ### 3. Warp the gradient of I with W(x;p) to compute \grad I
        Ix = spline_It1.ev(X_warp, Y_warp, dx=1)  # Gradient in x, 1st order derivative, shape: (36, 87)
        Iy = spline_It1.ev(X_warp, Y_warp, dy=1)  # Gradient in y, 1st order derivative, shape: (36, 87)

        ### 4. Evaluate the Jacobian dW/dp, Note: this case dW/dp = identity matrix

        ### 5. Compute the approximated Hessian H = \sum_{x} \grad I \grad I^T (A^T @ A)
        A = np.vstack((Ix.flatten(), Iy.flatten())).T # shape: (36 * 87, 2)
        H = A.T @ A # shape: (2, 2)

        ### 6. Computer delta p (dp) = H^{-1} @ A^T @ error
        dp = np.linalg.inv(H) @ A.T @ error.flatten()
        
        ### 7. Update warp parameters p <- p + dp
        p += dp

        # Terminate if the change in p (dp) is below the threshold
        if np.linalg.norm(dp) < threshold:
            break
    
    return p
