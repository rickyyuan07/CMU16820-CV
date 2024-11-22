# ##################################################################### #
# 16820: Computer Vision Homework 5
# Carnegie Mellon University
# Nov, 2023
###################################################################### #

import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2xyz
from utils import plotSurface, integrateFrankot
import cv2


def renderNDotLSphere(center, rad, light, pxSize, res):
    """
    Question 1 (b)

    Render a hemispherical bowl with a given center and radius. Assume that
    the hollow end of the bowl faces in the positive z direction, and the
    camera looks towards the hollow end in the negative z direction. The
    camera's sensor axes are aligned with the x- and y-axes.

    Parameters
    ----------
    center : numpy.ndarray
        The center of the hemispherical bowl in an array of size (3,)

    rad : float
        The radius of the bowl

    light : numpy.ndarray
        The direction of incoming light

    pxSize : float
        Pixel size

    res : numpy.ndarray
        The resolution of the camera frame

    Returns
    -------
    image : numpy.ndarray
        The rendered image of the hemispherical bowl
    """

    [X, Y] = np.meshgrid(np.arange(res[0]), np.arange(res[1]))
    X = (X - res[0] / 2) * pxSize * 1.0e-4
    Y = (Y - res[1] / 2) * pxSize * 1.0e-4
    Z = np.sqrt(rad**2 + 0j - X**2 - Y**2)
    X[np.real(Z) == 0] = 0
    Y[np.real(Z) == 0] = 0
    Z = np.real(Z)

    # Normals
    N = np.stack((X, Y, Z), axis=-1)  # Shape: (2160, 3840, 3)
    
    # n-dot-l shading
    image = np.maximum(0, np.sum(N * light, axis=-1))
    return image


def loadData(path="../data/"):
    """
    Question 1 (c)

    Load data from the path given. The images are stored as input_n.tif
    for n = {1...7}. The source lighting directions are stored in
    sources.mat.

    Parameters
    ---------
    path: str
        Path of the data directory

    Returns
    -------
    I : numpy.ndarray
        The 7 x P matrix of vectorized images

    L : numpy.ndarray
        The 3 x 7 matrix of lighting directions

    s: tuple
        Image shape

    """

    I = []
    L = np.load(path + "sources.npy").T  # (3, 7)

    for i in range(7):
        img = plt.imread(f"{path}input_{i+1}.tif", cv2.IMREAD_UNCHANGED)
        img = rgb2xyz(img)
        img = img[:, :, 1]  # 1 is the luminance channel
        s = img.shape
        img = img.flatten()
        I.append(img)
    
    I = np.array(I)  # (7, P)
    
    return I, L, s


def estimatePseudonormalsCalibrated(I, L):
    """
    Question 1 (e)

    In calibrated photometric stereo, estimate pseudonormals from the
    light direction and image matrices

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P array of vectorized images

    L : numpy.ndarray
        The 3 x 7 array of lighting directions

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals
    """

    # Solve the linear system using least squares: argmin_B ||L^T B - I||^2
    B = np.linalg.lstsq(L.T, I, rcond=None)[0]  # rcond=None uses default machine precision cutoff
    return B


def estimateAlbedosNormals(B):
    """
    Question 1 (e)

    From the estimated pseudonormals, estimate the albedos and normals

    Parameters
    ----------
    B : numpy.ndarray
        The 3 x P matrix of estimated pseudonormals

    Returns
    -------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals
    """

    # albedos are the magnitudes of the pseudonormals
    albedos = np.linalg.norm(B, axis=0)  # (P,)
    normals = B / (albedos + 1e-7)  # (3, P)
    return albedos, normals


def displayAlbedosNormals(albedos, normals, s):
    """
    Question 1 (f, g)

    From the estimated pseudonormals, display the albedo and normal maps

    Please make sure to use the `coolwarm` colormap for the albedo image
    and the `rainbow` colormap for the normals.

    Parameters
    ----------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    -------
    albedoIm : numpy.ndarray
        Albedo image of shape s

    normalIm : numpy.ndarray
        Normals reshaped as an s x 3 image

    """

    albedoIm = albedos.reshape(s)  # (2160, 3840)
    normals_rescaled = (normals + 1) / 2  # [-1, 1] -> [0, 1]
    normalIm = normals_rescaled.T.reshape((s[0], s[1], 3))  # (2160, 3840, 3)

    # # Display the images
    # plt.figure(figsize=(10, 5))
    
    # plt.subplot(1, 2, 1)
    # plt.imshow(albedoIm, cmap="gray")
    # plt.title("Albedo Map")
    # plt.axis("off")
    
    # plt.subplot(1, 2, 2)
    # plt.imshow(normalIm, cmap="rainbow")
    # plt.title("Normal Map")
    # plt.axis("off")
    
    # plt.show()
    return albedoIm, normalIm


def estimateShape(normals, s):
    """
    Question 1 (j)

    Integrate the estimated normals to get an estimate of the depth map
    of the surface.

    Parameters
    ----------
    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    ----------
    surface: numpy.ndarray
        The image, of size s, of estimated depths at each point

    """

    normals = normals.T.reshape((s[0], s[1], 3))  # (2160, 3840, 3)
    fx = -1.0 * normals[:, :, 0] / normals[:, :, 2]  # dz/dx
    fy = -1.0 * normals[:, :, 1] / normals[:, :, 2]  # dz/dy
    
    surface = integrateFrankot(fx, fy)
    return surface


if __name__ == "__main__":
    # Part 1(b)
    radius = 0.75  # cm
    center = np.asarray([0, 0, 0])  # cm
    pxSize = 7  # um
    res = (3840, 2160)

    light = np.asarray([1, 1, 1]) / np.sqrt(3)
    image = renderNDotLSphere(center, radius, light, pxSize, res)
    plt.figure()
    plt.imshow(image, cmap="gray")
    plt.imsave("1b-a.png", image, cmap="gray")

    light = np.asarray([1, -1, 1]) / np.sqrt(3)
    image = renderNDotLSphere(center, radius, light, pxSize, res)
    plt.figure()
    plt.imshow(image, cmap="gray")
    plt.imsave("1b-b.png", image, cmap="gray")

    light = np.asarray([-1, -1, 1]) / np.sqrt(3)
    image = renderNDotLSphere(center, radius, light, pxSize, res)
    plt.figure()
    plt.imshow(image, cmap="gray")
    plt.imsave("1b-c.png", image, cmap="gray")

    # Part 1(c)
    I, L, s = loadData("../data/")

    # Part 1(d)
    U, S, V = np.linalg.svd(I, full_matrices=False)
    print("Singular values:", S)
    rank = np.sum(S > 1e-5)
    print(f"Estimated rank of I: {rank}")

    # Part 1(e)
    B = estimatePseudonormalsCalibrated(I, L)

    # Part 1(f)
    albedos, normals = estimateAlbedosNormals(B)
    albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)
    plt.imsave("1f-a.png", albedoIm, cmap="gray")
    plt.imsave("1f-b.png", normalIm, cmap="rainbow")

    # Part 1(i)
    surface = estimateShape(normals, s)
    plotSurface(surface)
