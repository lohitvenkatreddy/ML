import numpy as np
import matplotlib.pyplot as plt

# --------- From A1 (reuse) ---------
def summation_unit(inputs, weights, bias=0.0):
    return np.dot(inputs, weights) + bias

def step_activation(x):
    return 1 if x >= 0 else 0

def bipolar_step_activation(x):
    return 1 if x >= 0 else -1

def sigmoid_activation(x):
    return 1 / (1 + np.exp(-x))

def relu_activation(x):
    return np.maximum(0, x)

def error_comparator(y_true, y_pred):
    return y_true - y_pred


# --------- Generic Perceptron Training ---------
def train_perceptron(X, y, activation_fn, learning_rate=0.05, max_epochs=1000, tol=0.002):
    weights = np.array([10.0, 0.2, -0.75])  # [bias, w1, w2]

    errors_per_epoch = []
    n_samples = X.shape[0]

    for epoch in range(max_epochs):
        total_error = 0.0
        for i in range(n_samples):
            xi = X[i]
            target = y[i]

            # Net input
            net_input = summation_unit(xi, weights[1:], weights[0])

            # Activation
            output = activation_fn(net_input)

            # Error
            error = error_comparator(target, output)
            total_error += error ** 2

            # Update weights
            weights[1:] += learning_rate * error * xi
            weights[0] += learning_rate * error

        errors_per_epoch.append(total_error)

        if total_error <= tol:
            break

    return weights, errors_per_epoch, epoch + 1


# --------- Main ---------
if __name__ == "__main__":
    # XOR Gate dataset
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    y_step = np.array([0, 1, 1, 0])        # For step/relu/sigmoid
    y_bipolar = np.array([0, 1, 1, 0])     # Bipolar also set to XOR truth table

    results = {}

    # Step
    w_step, e_step, epochs_step = train_perceptron(X, y_step, step_activation)
    results["Step"] = (epochs_step, e_step)

    # Bipolar Step
    w_bipolar, e_bipolar, epochs_bipolar = train_perceptron(X, y_bipolar, bipolar_step_activation)
    results["Bipolar Step"] = (epochs_bipolar, e_bipolar)

    # Sigmoid
    w_sigmoid, e_sigmoid, epochs_sigmoid = train_perceptron(X, y_step, sigmoid_activation)
    results["Sigmoid"] = (epochs_sigmoid, e_sigmoid)

    # ReLU
    w_relu, e_relu, epochs_relu = train_perceptron(X, y_step, relu_activation)
    results["ReLU"] = (epochs_relu, e_relu)

    # Print results
    for act, (ep, _) in results.items():
        print(f"{act}: Training stopped after {ep} epochs")

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(e_step) + 1), e_step, marker='o', label="Step")
    plt.plot(range(1, len(e_bipolar) + 1), e_bipolar, marker='x', label="Bipolar Step")
    plt.plot(range(1, len(e_sigmoid) + 1), e_sigmoid, marker='s', label="Sigmoid")
    plt.plot(range(1, len(e_relu) + 1), e_relu, marker='d', label="ReLU")

    plt.title("Epochs vs Error (XOR Gate, Different Activations)")
    plt.xlabel("Epochs")
    plt.ylabel("Sum-Squared Error")
    plt.legend()
    plt.grid(True)
    plt.show()
