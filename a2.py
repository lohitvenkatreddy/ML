import numpy as np
import matplotlib.pyplot as plt

# --------- From A1 (reuse) ---------
def summation_unit(inputs, weights, bias=0.0):
    return np.dot(inputs, weights) + bias

def step_activation(x):
    return 1 if x >= 0 else 0

def error_comparator(y_true, y_pred):
    return y_true - y_pred


# --------- A2 Perceptron Training ---------
def train_perceptron(X, y, learning_rate=0.05, max_epochs=1000, tol=0.002):
    # Initialize weights (bias + two inputs)
    weights = np.array([10.0, 0.2, -0.75])  # [bias, w1, w2]

    errors_per_epoch = []
    n_samples = X.shape[0]

    for epoch in range(max_epochs):
        total_error = 0.0
        for i in range(n_samples):
            xi = X[i]
            target = y[i]

            # Net input using A1 summation
            net_input = summation_unit(xi, weights[1:], weights[0])

            # Activation using A1
            output = step_activation(net_input)

            # Error using A1
            error = error_comparator(target, output)
            total_error += error ** 2

            # Update weights
            weights[1:] += learning_rate * error * xi
            weights[0] += learning_rate * error  # bias update

        errors_per_epoch.append(total_error)

        if total_error <= tol:
            break

    return weights, errors_per_epoch, epoch + 1


# --------- Main ---------
if __name__ == "__main__":
    # AND Gate dataset
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    y = np.array([0, 0, 0, 1])

    # Train perceptron
    final_weights, errors, epochs_taken = train_perceptron(X, y)

    # Results
    print("Final Weights:", final_weights)
    print("Epochs Taken:", epochs_taken)

    # Plot epochs vs error
    plt.plot(range(1, len(errors) + 1), errors, marker='o')
    plt.title("Epochs vs Error (AND Gate)")
    plt.xlabel("Epochs")
    plt.ylabel("Sum-Squared Error")
    plt.grid(True)
    plt.show()
