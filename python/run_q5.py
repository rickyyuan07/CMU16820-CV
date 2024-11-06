import numpy as np
import scipy.io
from nn import *
from collections import Counter
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import string

np.random.seed(42)
train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

max_iters = 100
# pick a batch size, learning rate
batch_size = 36 
learning_rate =  3e-5
hidden_size = 32
lr_rate = 20
batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)

params = Counter()

# Q5.1 & Q5.2
# Initialize the layers for the autoencoder
input_size = train_x.shape[1]  # 1024 for the NIST36 dataset
initialize_weights(input_size, hidden_size, params, "layer1")
initialize_weights(hidden_size, hidden_size, params, "layer2")
initialize_weights(hidden_size, hidden_size, params, "layer3")
initialize_weights(hidden_size, input_size, params, "output")
# Initialize the momentum terms
for k in ["layer1", "layer2", "layer3", "output"]:
    params["m_W" + k] = np.zeros_like(params["W" + k])
    params["m_b" + k] = np.zeros_like(params["b" + k])

# should look like your previous training loops
losses = []
for itr in range(max_iters):
    total_loss = 0
    for xb,_ in batches:
        # training loop can be exactly the same as q2!
        # your loss is now squared error
        # delta is the d/dx of (x-y)^2
        # to implement momentum
        #   use 'm_'+name variables in initialize_weights from nn.py
        #   to keep a saved value
        #   params is a Counter(), which returns a 0 if an element is missing
        #   so you should be able to write your loop without any special conditions

        # forward pass
        h1 = forward(xb, params, "layer1", relu)
        h2 = forward(h1, params, "layer2", relu)
        h3 = forward(h2, params, "layer3", relu)
        reconstructed_x = forward(h3, params, "output", sigmoid)

        # loss
        loss = np.sum((xb - reconstructed_x) ** 2) # total squared error
        total_loss += loss

        # backward
        delta = 2 * (reconstructed_x - xb)
        delta = backwards(delta, params, "output", sigmoid_deriv)
        delta = backwards(delta, params, "layer3", relu_deriv)
        delta = backwards(delta, params, "layer2", relu_deriv)
        backwards(delta, params, "layer1", relu_deriv)

        # apply gradient, remember to update momentum as well
        for k in params.keys(): # Copy key list to avoid adding new keys during iteration
            if "grad" in k:
                layer_name = k.split('_')[1]  # Layer name, e.g. Wlayer1
                m_key = "m_" + layer_name  # Momentum key, e.g. m_Wlayer1
                params[m_key] = 0.9 * params[m_key] - learning_rate * params[k]
                params[layer_name] += params[m_key]
    
    losses.append(total_loss/train_x.shape[0])
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr,total_loss))
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9

# plot loss curve
plt.plot(range(len(losses)), losses)
plt.xlabel("epoch")
plt.ylabel("average loss")
plt.xlim(0, len(losses)-1)
plt.ylim(0, None)
plt.grid()
plt.show()

        
# Q5.3.1
# choose 5 labels (change if you want)
visualize_labels = ["R", "I", "C", "K", "Y"]

# get 2 validation images from each label to visualize
visualize_x = np.zeros((2*len(visualize_labels), valid_x.shape[1])) # (10, 1024)
for i, label in enumerate(visualize_labels):
    idx = 26+int(label) if label.isnumeric() else string.ascii_lowercase.index(label.lower())
    choices = np.random.choice(np.arange(100*idx, 100*(idx+1)), 2, replace=False)
    visualize_x[2*i:2*i+2] = valid_x[choices]

# run visualize_x through your network
# name the output reconstructed_x
h1 = forward(visualize_x, params, "layer1", relu)
h2 = forward(h1, params, "layer2", relu)
h3 = forward(h2, params, "layer3", relu)
reconstructed_x = forward(h3, params, "output", sigmoid)

# visualize
fig = plt.figure()
plt.axis("off")
grid = ImageGrid(fig, 111, nrows_ncols=(len(visualize_labels), 4), axes_pad=0.05)
for i, ax in enumerate(grid):
    if i % 2 == 0:
        ax.imshow(visualize_x[i//2].reshape((32, 32)).T, cmap="Greys")
    else:
        ax.imshow(reconstructed_x[i//2].reshape((32, 32)).T, cmap="Greys")
    ax.set_axis_off()
plt.show()


# Q5.3.2
from skimage.metrics import peak_signal_noise_ratio
# evaluate PSNR
h1 = forward(valid_x, params, "layer1", relu)
h2 = forward(h1, params, "layer2", relu)
h3 = forward(h2, params, "layer3", relu)
pred_valid_x = forward(h3, params, "output", sigmoid)

psnr_values = []
for original, reconstructed in zip(valid_x, pred_valid_x):
    psnr = peak_signal_noise_ratio(original, reconstructed)
    psnr_values.append(psnr)

average_psnr = np.mean(psnr_values)
print(f"Average PSNR: {average_psnr:.2f}")
