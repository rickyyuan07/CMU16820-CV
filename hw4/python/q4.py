import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation


# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes
    # this can be 10 to 15 lines of code using skimage functions

    # Denoise the image using a denoising filter
    # Ref: https://scikit-image.org/docs/stable/auto_examples/filters/plot_denoise.html
    # https://medium.com/@betulmesci/image-processing-tutorial-using-scikit-image-noise-20e181c541a1
    denoised = skimage.restoration.denoise_bilateral(image, channel_axis=-1)
    # denoised = skimage.filters.gaussian(image, sigma=1)

    # 2. Convert to grayscale
    gray = skimage.color.rgb2gray(denoised)

    # 3. Apply a threshold to get a binary (black-and-white) image
    threshold = skimage.filters.threshold_otsu(gray)
    bw = gray < threshold  # Invert the binary image if needed

    # 4. Apply morphological operations to clean up small noise
    bw = skimage.morphology.remove_small_objects(bw, min_size=100)
    bw = skimage.morphology.closing(bw, skimage.morphology.square(6))

    # 5. Label connected components, 8-connectivity
    labeled = skimage.measure.label(bw, connectivity=2)

    # 6. Find regions and filter small components (bounding boxes)
    regions = skimage.measure.regionprops(labeled)

    # 7. Filter regions based on area
    areas = [region.area for region in regions]
    mean, std = np.mean(areas), np.std(areas)
    # Drop all regions with area less than mean - 2.5 * std
    bboxes = [region.bbox for region in regions if region.area > mean - 2.5 * std]

    # Make the stroke of the letters thicker
    bw = skimage.morphology.dilation(bw, footprint=skimage.morphology.square(5))
    return bboxes, bw.astype(np.float64)
