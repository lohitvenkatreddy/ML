import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

def generate_training_data():
    X = np.random.uniform(1, 10, (20, 2))
    y = np.array([0 if sum(point) < 12 else 1 for point in X])
    return X, y

def generate_test_grid():
    grid_x, grid_y = np.meshgrid(np.arange(0, 10.1, 0.1), np.arange(0, 10.1, 0.1))
    grid_points = np.c_[grid_x.ravel(), grid_y.ravel()]
    return grid_points

# ==== MAIN ====
X_train, y_train = generate_training_data()
X_test = generate_test_grid()

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

plt.figure(figsize=(8, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='bwr', alpha=0.3, s=5)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr', edgecolor='k', s=50)
plt.title("A4: kNN Prediction on Test Grid (k=3)")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()
