import numpy as np
import matplotlib.pyplot as plt

def generate_data():
    X = np.random.uniform(1, 10, (20, 2))
    y = np.array([0 if sum(point) < 12 else 1 for point in X])
    return X, y

# ==== MAIN ====
X, y = generate_data()

plt.figure(figsize=(8, 6))
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='blue', label='Class 0 (Blue)')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='red', label='Class 1 (Red)')
plt.title("A3: Training Data Classification")
plt.xlabel("Feature X")
plt.ylabel("Feature Y")
plt.legend()
plt.grid(True)
plt.show()
