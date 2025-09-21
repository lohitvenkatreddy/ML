import numpy as np

# Dataset: Candies, Mangoes, Milk Packets, and Payment
X = np.array([
    [20, 6, 2],  # C_1
    [16, 3, 6],  # C_2
    [27, 6, 2],  # C_3
    [19, 1, 2],  # C_4
    [24, 4, 2],  # C_5
    [22, 1, 5],  # C_6
    [15, 4, 2],  # C_7
    [18, 4, 2],  # C_8
    [21, 1, 4],  # C_9
    [16, 2, 4],  # C_10
])

# Output: High Value Transaction (Yes = 1, No = 0)
y = np.array([1, 1, 1, 0, 1, 0, 1, 1, 0, 0])

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Sigmoid derivative
def sigmoid_derivative(x):
    return x * (1 - x)

# Initialize weights and bias
np.random.seed(42)
weights = np.random.rand(X.shape[1])
bias = np.random.rand(1)

# Hyperparameters
learning_rate = 0.05
epochs = 10000

# Perceptron Learning Rule
for epoch in range(epochs):
    # Linear combination
    linear_output = np.dot(X, weights) + bias
    
    # Apply sigmoid activation
    output = sigmoid(linear_output)
    
    # Calculate error
    error = y - output
    
    # Update weights and bias
    weights += learning_rate * np.dot(X.T, error * sigmoid_derivative(output))
    bias += learning_rate * np.sum(error * sigmoid_derivative(output))
    
    # Printing error for each 1000th epoch
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}: Error {np.mean(np.abs(error))}")

# Final weights and bias
print("\nFinal Weights:", weights)
print("Final Bias:", bias)

# Final predictions
predictions = np.round(sigmoid(np.dot(X, weights) + bias))
print("\nPredictions:", predictions)
