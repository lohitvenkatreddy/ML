import numpy as np

from a6 import sigmoid

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

# Adding bias term to X (X_0 = 1)
X_biased = np.c_[np.ones(X.shape[0]), X]

# Using Pseudo-Inverse to calculate weights
weights = np.linalg.pinv(X_biased).dot(y)

# Making predictions
predictions = np.round(sigmoid(X_biased.dot(weights)))
print("\nWeights from Pseudo-Inverse:", weights)
print("Predictions from Pseudo-Inverse:", predictions)

# Calculate error
error = np.mean(np.abs(y - predictions))
print("\nError using Matrix Pseudo-Inverse:", error)
