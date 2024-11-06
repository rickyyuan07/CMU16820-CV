import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *

# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

GTS = {
    "01_list.jpg": ["TODOLIST", "1MAKEATODOLIST", "2CHECKOFFTHEFIRST",
                    "THIHGONTODOLIST", "3REALIZEYOUHAVEALREADY", "COMPLETED2THINGS",
                    "4REWARDYOURSELFWITH", "ANAP"],
    "02_letters.jpg": ["ABCDEFG", "HIJKLMN", "OPQRSTU", "VWXYZ", "1234567890"],
    "03_haiku.jpg": ["HAIKUSAREEASY", "BUTSOMETIMESTHEYDONTMAKESENSE",
                     "REFRIGERATOR"],
    "04_deep.jpg": ["DEEPLEARNING", "DEEPERLEARNING", "DEEPESTLEARNING"]
}

tottot_acc = 0
tottot = 0
for img in os.listdir("../images"):
    print(f"Processing {img}")
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join("../images", img)))
    bboxes, bw = findLetters(im1)

    plt.imshow(bw)
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle(
            (minc, minr),
            maxc - minc,
            maxr - minr,
            fill=False,
            edgecolor="red",
            linewidth=2,
        )
        plt.gca().add_patch(rect)
    # plt.show()
    plt.savefig(f"../Figures/4_3_{img.split('_')[0]}_bboxes.png")
    plt.clf()
    # find the rows using..RANSAC, counting, clustering, etc.
    ##########################
    ##### your code here #####
    ##########################
    bboxes.sort(key=lambda x: (x[0]+x[2])/2)  # Sort boxes by the center y (row)

    rows = [] # list of rows, each row is a list of bounding boxes
    cur_row = [bboxes[0]]
    ROW_THRESHOLD = 150
    if img == "01_list.jpg":
        ROW_THRESHOLD = 100
    for bbox in bboxes[1:]:
        center_row = (bbox[0] + bbox[2]) / 2
        cur_row_center = (cur_row[0][0] + cur_row[0][2]) / 2
        if abs(center_row - cur_row_center) < ROW_THRESHOLD: # same row
            cur_row.append(bbox)
        else: # new row
            rows.append(cur_row)
            cur_row = [bbox]

    rows.append(cur_row) # add the last row

    for row in rows:
        row.sort(key=lambda x: (x[1]+x[3]/2)) # sort boxes in each row by the center x (column)

    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    ##########################
    ##### your code here #####
    ##########################

    cropped_images = [] # list of lists of cropped images from rows

    for bboxes in rows:
        cropped_image = []
        for bbox in bboxes:
            minr, minc, maxr, maxc = bbox
            # Crop the region and pad to make it square
            cropped = bw[minr:maxr, minc:maxc]
            # Make the crop square by padding
            height, width = cropped.shape
            if height > width: # pad width
                pad_width = (height - width) // 2
                cropped = np.pad(cropped, ((0, 0), (pad_width, pad_width)), mode='constant', constant_values=0)
            else: # pad height
                pad_height = (width - height) // 2
                cropped = np.pad(cropped, ((pad_height, pad_height), (0, 0)), mode='constant', constant_values=0)
            # Resize to 32x32 and add to the list
            cropped = skimage.transform.resize(cropped, (24, 24))
            cropped = np.pad(cropped, ((4,4), (4, 4)), mode='constant', constant_values=0)
            if img == "04_deep.jpg" or img == "02_letters.jpg":
                cropped = skimage.morphology.dilation(cropped, footprint=skimage.morphology.square(2))
            elif img == "01_list.jpg":
                cropped = skimage.morphology.erosion(cropped, footprint=skimage.morphology.square(1))

            cropped_image.append(cropped)
            
        cropped_images.append(cropped_image)

    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string

    letters = np.array(
        [_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)]
    )
    params = pickle.load(open("q3_weights.pickle", "rb"))
    ##########################
    ##### your code here #####
    ##########################

    import scipy.io
    train_data = scipy.io.loadmat("../data/nist36_train.mat")
    valid_data = scipy.io.loadmat("../data/nist36_valid.mat")
    test_data = scipy.io.loadmat("../data/nist36_test.mat")

    train_x, train_y = train_data["train_data"], train_data["train_labels"]
    valid_x, valid_y = valid_data["valid_data"], valid_data["valid_labels"]
    test_x, test_y = test_data["test_data"], test_data["test_labels"]

    LABEL = GTS[img]
    total_char = sum([len(label) for label in LABEL])
    acc = 0
    tottot += total_char
    # Classify each cropped image
    for i, cropped_image in enumerate(cropped_images):
        print(f"Row {i+1}: ", end="\n")
        pred = ""
        for cropped in cropped_image:
            cropped = 1.0 - cropped
            input_image = cropped.T.reshape(1, -1)
            h1 = forward(input_image, params, "layer1")
            probs = forward(h1, params, "output", softmax)
            predicted_label = letters[np.argmax(probs)]
            pred += predicted_label

        print(f"{pred}")
        print(f"{LABEL[i]}")

        try:
            acc += sum([1 for j in range(len(pred)) if pred[j] == LABEL[i][j]])
        except:
            breakpoint()
        
    tottot_acc += acc
    print(f"\nAccuracy: {acc}/{total_char} ({acc/total_char*100:.2f}%)")

print(f"Total Accuracy: {tottot_acc}/{tottot} ({tottot_acc/tottot*100:.2f}%)")