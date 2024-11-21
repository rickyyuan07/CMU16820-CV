# ##################################################################### #
# 16820: Computer Vision Homework 5
# Carnegie Mellon University
# 
# Nov, 2023
# ##################################################################### #

import numpy as np
import matplotlib.pyplot as plt
from q1 import (
    loadData,
    estimateAlbedosNormals,
    displayAlbedosNormals,
    estimateShape,
)
from q1 import estimateShape
from utils import enforceIntegrability, plotSurface

def estimatePseudonormalsUncalibrated(I):
    """
    Question 2 (b)

    Estimate pseudonormals without the help of light source directions.

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P matrix of loaded images

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pseudonormals

    L : numpy.ndarray
        The 3 x 7 array of lighting directions

    """

    # (7, P) -> (7, 7), (7,), (7, P)
    U, S, V = np.linalg.svd(I, full_matrices=False)
    B = V[:3, :]  # (3, P)
    L = U[:, :3]  # (7, 3)
    return B, L.T


def plotBasRelief(B, mu, nu, lam):
    """
    Question 2 (f)

    Make a 3D plot of of a bas-relief transformation with the given parameters.

    Parameters
    ----------
    B : numpy.ndarray
        The 3 x P matrix of pseudonormals

    mu : float
        bas-relief parameter

    nu : float
        bas-relief parameter

    lambda : float
        bas-relief parameter

    Returns
    -------
        None

    """
    G = np.array([[1, 0, 0], [0, 1, 0], [mu, nu, lam]])
    B = enforceIntegrability(B, s)
    B = np.linalg.inv(G.T) @ B
    albedos, normals = estimateAlbedosNormals(B)
    surface = estimateShape(normals, s)
    plotSurface(surface)


if __name__ == "__main__":
    I, L0, s = loadData()
    B0, L = estimatePseudonormalsUncalibrated(I)

    # Part 2 (b)
    albedos, normals = estimateAlbedosNormals(B0)
    albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)
    plt.imsave("2b-a.png", albedoIm, cmap="gray")
    plt.imsave("2b-b.png", normalIm, cmap="rainbow")

    # Part 2 (c)
    print(L)
    print(L0)

    # Part 2 (d)
    surface = estimateShape(normals, s)
    plotSurface(surface)

    # Part 2 (e)
    # GBR (generalized bas-relief) transform matrix
    # G = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    plotBasRelief(B0, 0, 0, -1)

    # Part 2 (f)
    # Compare different mu
    plotBasRelief(B0, 3, 0, -1)
    plotBasRelief(B0, -3, 0, -1)
    plotBasRelief(B0, 10, 0, -1)

    # Compare different nu
    plotBasRelief(B0, 0, 3, -1)
    plotBasRelief(B0, 0, -3, -1)
    plotBasRelief(B0, 0, 10, -1)

    # Compare different lambda
    plotBasRelief(B0, 0, 0, -1)
    plotBasRelief(B0, 0, 0, -10)
    plotBasRelief(B0, 0, 0, -50)
