import numpy as np

# ---------- A1 Implementation ----------

# a) Summation Unit
def summation_unit(inputs, weights, bias=0.0):
    """
    Computes weighted sum of inputs + bias
    inputs: list or numpy array of input values
    weights: list or numpy array of weights
    bias: bias term (default=0.0)
    """
    return np.dot(inputs, weights) + bias


# b) Activation Functions
def step_activation(x):
    return 1 if x >= 0 else 0

def bipolar_step_activation(x):
    return 1 if x >= 0 else -1

def sigmoid_activation(x):
    return 1 / (1 + np.exp(-x))

def tanh_activation(x):
    return np.tanh(x)

def relu_activation(x):
    return np.maximum(0, x)

def leaky_relu_activation(x, alpha=0.01):
    return x if x > 0 else alpha * x


# c) Comparator unit for Error calculation
def error_comparator(y_true, y_pred):
    """
    Returns difference between true and predicted output
    """
    return y_true - y_pred


# ----------- Main (Test Section) -----------
if __name__ == "__main__":
    # Example inputs
    inputs = np.array([1, 0])
    weights = np.array([0.5, -0.6])
    bias = 0.2

    # Summation
    net_input = summation_unit(inputs, weights, bias)
    print("Summation Output:", net_input)

    # Activations
    print("Step:", step_activation(net_input))
    print("Bipolar Step:", bipolar_step_activation(net_input))
    print("Sigmoid:", sigmoid_activation(net_input))
    print("Tanh:", tanh_activation(net_input))
    print("ReLU:", relu_activation(net_input))
    print("Leaky ReLU:", leaky_relu_activation(net_input))

    # Error Comparator Example
    y_true, y_pred = 1, sigmoid_activation(net_input)
    print("Error:", error_comparator(y_true, y_pred))
