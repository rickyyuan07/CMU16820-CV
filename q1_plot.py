import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Define sigmoid function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Define tanh function and its derivative
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

# Generate values for x
x = np.linspace(-6, 6, 100)

# Compute values for sigmoid, derivative of sigmoid, tanh, and derivative of tanh
sigmoid_values = sigmoid(x)
sigmoid_derivative_values = sigmoid_derivative(x)
tanh_values = tanh(x)
tanh_derivative_values = tanh_derivative(x)

# Plot the functions
plt.figure(figsize=(12, 6))

# Plot sigmoid and its derivative
plt.subplot(1, 2, 1)
plt.plot(x, sigmoid_values, label="Sigmoid", color="blue")
plt.plot(x, sigmoid_derivative_values, label="Sigmoid Derivative", color="red", linestyle="--")
plt.title("Sigmoid and Its Derivative")
plt.xlabel("x")
plt.ylabel("Value")
plt.legend()
# 增加主格線和次格線
plt.grid(which='major', linestyle='-', linewidth=0.8)
plt.grid(which='minor', linestyle='--', linewidth=0.5)

# 設置 y 軸的次格線間隔為 0.5
plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(0.1))

# Plot tanh and its derivative
plt.subplot(1, 2, 2)
plt.plot(x, tanh_values, label="Tanh", color="green")
plt.plot(x, tanh_derivative_values, label="Tanh Derivative", color="orange", linestyle="--")
plt.title("Tanh and Its Derivative")
plt.xlabel("x")
plt.ylabel("Value")
plt.legend()
# 增加主格線和次格線
plt.grid(which='major', linestyle='-', linewidth=0.8)
plt.grid(which='minor', linestyle='--', linewidth=0.5)

# 設置 y 軸的次格線間隔為 0.5
plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(0.5))

plt.tight_layout()
plt.show()