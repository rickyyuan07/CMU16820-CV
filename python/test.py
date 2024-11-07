import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from nn import *

train_data = scipy.io.loadmat("../data/nist36_train.mat")
valid_data = scipy.io.loadmat("../data/nist36_valid.mat")
test_data = scipy.io.loadmat("../data/nist36_test.mat")

train_x, train_y = train_data["train_data"], train_data["train_labels"]
valid_x, valid_y = valid_data["valid_data"], valid_data["valid_labels"]
test_x, test_y = test_data["test_data"], test_data["test_labels"]


import pickle
import string

letters = np.array(
    [_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)]
)
params = pickle.load(open("q3_weights.pickle", "rb"))

# Test the neural network on the test set
for i in range(test_x.shape[0]):
    i = np.random.randint(test_x.shape[0])
    x = test_x[i].reshape(1, -1)
    y = test_y[i]
    # breakpoint()
    h1 = forward(x, params, "layer1")
    probs = forward(h1, params, "output", softmax)
    pred = np.argmax(probs)
    actual = np.argmax(y)
    print(f"Predicted: {letters[pred]}, Actual: {letters[actual]}")
    plt.imshow(x.reshape(32, 32).T, cmap="Greys")
    plt.show()