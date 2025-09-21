import numpy as np

# AND gate input and output
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0], [0], [0], [1]])  # AND gate output

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Sigmoid derivative
def sigmoid_derivative(x):
    return x * (1 - x)

# Initialize weights and biases
np.random.seed(42)
input_layer_size = 2
hidden_layer_size = 2
output_layer_size = 1

# Random initialization of weights
weights_input_hidden = np.random.rand(input_layer_size, hidden_layer_size)
weights_hidden_output = np.random.rand(hidden_layer_size, output_layer_size)

# Bias initialization
bias_hidden = np.random.rand(1, hidden_layer_size)
bias_output = np.random.rand(1, output_layer_size)

# Learning rate
learning_rate = 0.05
epochs = 10000

# Backpropagation
for epoch in range(epochs):
    # Forward propagation
    hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)
    
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    output_layer_output = sigmoid(output_layer_input)
    
    # Compute error
    error = y - output_layer_output
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}: Error {np.mean(np.abs(error))}")
    
    # Backward propagation (delta calculation)
    output_layer_delta = error * sigmoid_derivative(output_layer_output)
    hidden_layer_delta = output_layer_delta.dot(weights_hidden_output.T) * sigmoid_derivative(hidden_layer_output)
    
    # Update weights and biases
    weights_hidden_output += hidden_layer_output.T.dot(output_layer_delta) * learning_rate
    weights_input_hidden += X.T.dot(hidden_layer_delta) * learning_rate
    bias_output += np.sum(output_layer_delta, axis=0, keepdims=True) * learning_rate
    bias_hidden += np.sum(hidden_layer_delta, axis=0, keepdims=True) * learning_rate

# Final output
print("\nPredictions after training:")
print(np.round(output_layer_output))
