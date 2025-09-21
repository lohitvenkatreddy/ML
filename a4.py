import numpy as np
import matplotlib.pyplot as plt

# --------- From A1 (reuse) ---------
def summation_unit(inputs, weights, bias=0.0):
    return np.dot(inputs, weights) + bias

def step_activation(x):
    return 1 if x >= 0 else 0

def error_comparator(y_true, y_pred):
    return y_true - y_pred


# --------- Perceptron Training ---------
def train_perceptron(X, y, learning_rate, max_epochs=1000, tol=0.002):
    weights = np.array([10.0, 0.2, -0.75])  # [bias, w1, w2]

    n_samples = X.shape[0]

    for epoch in range(max_epochs):
        total_error = 0.0
        for i in range(n_samples):
            xi = X[i]
            target = y[i]

            # Net input
            net_input = summation_unit(xi, weights[1:], weights[0])

            # Activation
            output = step_activation(net_input)

            # Error
            error = error_comparator(target, output)
            total_error += error ** 2

            # Update weights
            weights[1:] += learning_rate * error * xi
            weights[0] += learning_rate * error

        if total_error <= tol:
            return epoch + 1  # converged

    return max_epochs  # if not converged


# --------- Main ---------
if __name__ == "__main__":
    # AND Gate dataset
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    y = np.array([0, 0, 0, 1])

    learning_rates = [i / 10 for i in range(1, 11)]  # 0.1 to 1.0
    epochs_taken = []

    for lr in learning_rates:
        epochs = train_perceptron(X, y, learning_rate=lr)
        epochs_taken.append(epochs)
        print(f"Learning Rate {lr}: {epochs} epochs")

    # Plot learning rate vs epochs
    plt.plot(learning_rates, epochs_taken, marker='o')
    plt.title("Learning Rate vs Epochs to Converge (AND Gate)")
    plt.xlabel("Learning Rate")
    plt.ylabel("Epochs to Converge")
    plt.grid(True)
    plt.show()
