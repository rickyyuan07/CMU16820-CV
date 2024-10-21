import numpy as np
import matplotlib.pyplot as plt

from helper import _epipoles

from q2_1_eightpoint import eightpoint

# Insert your package here
from scipy.ndimage import gaussian_filter


# Helper functions for this assignment. DO NOT MODIFY!!!
def epipolarMatchGUI(I1, I2, F):
    e1, e2 = _epipoles(F)

    sy, sx, _ = I2.shape

    f, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 9))
    ax1.imshow(I1)
    ax1.set_title("Select a point in this image")
    ax1.set_axis_off()
    ax2.imshow(I2)
    ax2.set_title(
        "Verify that the corresponding point \n is on the epipolar line in this image"
    )
    ax2.set_axis_off()

    while True:
        plt.sca(ax1)
        # x, y = plt.ginput(1, mouse_stop=2)[0]

        out = plt.ginput(1, timeout=3600, mouse_stop=2)

        if len(out) == 0:
            print(f"Closing GUI")
            break

        x, y = out[0]

        xc = int(x)
        yc = int(y)
        v = np.array([xc, yc, 1])
        l = F.dot(v)
        s = np.sqrt(l[0] ** 2 + l[1] ** 2)

        if s == 0:
            print("Zero line vector in displayEpipolar")

        l = l / s

        if l[0] != 0:
            ye = sy - 1
            ys = 0
            xe = -(l[1] * ye + l[2]) / l[0]
            xs = -(l[1] * ys + l[2]) / l[0]
        else:
            xe = sx - 1
            xs = 0
            ye = -(l[0] * xe + l[2]) / l[1]
            ys = -(l[0] * xs + l[2]) / l[1]

        # plt.plot(x,y, '*', 'MarkerSize', 6, 'LineWidth', 2);
        ax1.plot(x, y, "*", markersize=6, linewidth=2)
        ax2.plot([xs, xe], [ys, ye], linewidth=2)

        # draw points
        x2, y2 = epipolarCorrespondence(I1, I2, F, xc, yc)
        ax2.plot(x2, y2, "ro", markersize=8, linewidth=2)
        plt.draw()


"""
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2
            
    Hints:
    (1) Given input [x1, x2], use the fundamental matrix to recover the corresponding epipolar line on image2
    (2) Search along this line to check nearby pixel intensity (you can define a search window) to 
        find the best matches
    (3) Use guassian weighting to weight the pixel simlairty

"""


def epipolarCorrespondence(im1, im2, F, x1: int, y1: int):
    window_size = 20
    # Assume two images only differ by a small amount, i.e. d(p1, p2) is small
    search_range = 50
    half_window = window_size // 2

    # Template patch around the point (x1, y1) in im1
    window1 = im1[y1 - half_window:y1 + half_window + 1, x1 - half_window:x1 + half_window + 1]
    
    # Gaussian filter helps us focus more on the center point
    window1 = gaussian_filter(window1, sigma=1)

    # Compute the epipolar line in im2 using fundamental matrix
    epipolar_line = F @ np.array([x1, y1, 1]).T
    epipolar_line = epipolar_line / np.linalg.norm(epipolar_line[:2])

    height, width = im2.shape[:2]
    min_y, max_y = max(0, y1 - search_range), min(height, y1 + search_range)

    best_match = None
    min_error = float('inf')
    # Iterate along the epipolar line within the range to find the best match
    for y2 in range(min_y, max_y):
        # x * l[0] + y * l[1] + l[2] = 0 ==> x = -(l[1] * y + l[2]) / l[0]
        x2 = int(-(epipolar_line[1] * y2 + epipolar_line[2]) / epipolar_line[0])
        if x2 - half_window < 0 or x2 + half_window >= width or y2 - half_window < 0 or y2 + half_window >= height:
            continue

        window2 = im2[y2 - half_window:y2 + half_window + 1, x2 - half_window:x2 + half_window + 1]
        # Same filter for the im2 window
        window2 = gaussian_filter(window2, sigma=1)

        error = np.sum((window1 - window2) ** 2)
        if error < min_error:
            min_error = error
            best_match = (x2, y2)

    return best_match


if __name__ == "__main__":
    correspondence = np.load("data/some_corresp.npz")  # Loading correspondences
    intrinsics = np.load("data/intrinsics.npz")  # Loading the intrinscis of the camera
    K1, K2 = intrinsics["K1"], intrinsics["K2"]
    pts1, pts2 = correspondence["pts1"], correspondence["pts2"]
    im1 = plt.imread("data/im1.png")
    im2 = plt.imread("data/im2.png")

    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))

    np.savez("q4_1.npz", F, pts1, pts2)
    epipolarMatchGUI(im1, im2, F)

    # Simple Tests to verify your implementation:
    x2, y2 = epipolarCorrespondence(im1, im2, F, 119, 217)
    assert np.linalg.norm(np.array([x2, y2]) - np.array([118, 181])) < 10
