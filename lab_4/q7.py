import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

def generate_data():
    X = np.random.uniform(1, 10, (50, 2))
    y = np.array([0 if sum(p) < 12 else 1 for p in X])
    return X, y

def find_best_k(X, y):
    param_grid = {'n_neighbors': list(range(1, 21))}
    grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
    grid.fit(X, y)
    return grid.best_params_, grid.best_score_

# ==== MAIN ====
X, y = generate_data()
best_params, best_score = find_best_k(X, y)

print("A7: Grid Search for Best k")
print("Best k:", best_params['n_neighbors'])
print("Best Cross-Validation Score:", best_score)
