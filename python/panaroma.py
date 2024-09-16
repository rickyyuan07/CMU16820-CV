import numpy as np
import cv2
from opts import get_opts
from HarryPotterize import warpImage

def crop_black_borders(img):
    # Crop the right black area
    for i in range(img.shape[1] - 1, 0, -1):
        if np.sum(img[:, i]) > 0:
            img = img[:, :i]
            break

    return img

def create_panorama(left_img, right_img, opts):
    width = left_img.shape[1] + right_img.shape[1]
    height = left_img.shape[0]

    panorama = np.zeros((height, width, 3), dtype=np.uint8)
    panorama[0:left_img.shape[0], 0:left_img.shape[1]] = left_img
    panorama = warpImage(opts, right_img, right_img, panorama)
    panorama = crop_black_borders(panorama)

    return panorama

if __name__ == "__main__":
    opts = get_opts()

    img1 = cv2.imread(opts.pano_left_img)
    img2 = cv2.imread(opts.pano_right_img)

    panorama = create_panorama(img1, img2, opts)

    # Save and display the result
    cv2.imwrite(opts.pano_out_img, panorama)
    cv2.imshow('Panorama', panorama)
    cv2.waitKey(0)
    cv2.destroyAllWindows()