import numpy as np
from scipy.ndimage import binary_erosion
from scipy.ndimage import binary_dilation
from scipy.ndimage import affine_transform
from LucasKanadeAffine import LucasKanadeAffine
from InverseCompositionAffine import InverseCompositionAffine

def SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance):
    """
    :param image1: Images at time t
    :param image2: Images at time t+1
    :param threshold: used for LucasKanadeAffine
    :param num_iters: used for LucasKanadeAffine
    :param tolerance: binary threshold of intensity difference when computing the mask
    :return: mask: [nxm]
    """

    ################### TODO Implement Substract Dominent Motion ###################
    mask = np.ones(image1.shape, dtype=bool)
    
    # Estimate dominant motion
    # M = LucasKanadeAffine(image1, image2, threshold, num_iters)
    M = InverseCompositionAffine(image1, image2, threshold, num_iters)
    
    # Warp image2 according to the estimated motion
    warped_image2 = affine_transform(image2, M)
    diff = np.abs(image1 - warped_image2)
    
    # True if considered as moving objects
    mask = (diff > tolerance).astype(bool)
    
    # Opening operation to remove noise
    mask = binary_erosion(mask, iterations=2, brute_force=True)
    mask = binary_dilation(mask, iterations=2, brute_force=True)

    return mask.astype(bool)
