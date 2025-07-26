import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Function to train and evaluate model accuracy for various k
def accuracy_for_k_range(X_train, X_test, y_train, y_test, k_range):
    accuracies = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        accuracy = knn.score(X_test, y_test)
        accuracies.append(accuracy)
    return accuracies

# Load and preprocess the dataset
df = pd.read_csv("Hyderbad_House_price.csv")
df['bhk'] = df['title'].str.extract(r'(\d) BHK').astype(float)
df = df[df['bhk'].isin([2, 3])]  # Binary classification

# Select features and labels
features = ['price(L)', 'rate_persqft', 'area_insqft']
X = df[features].values
y = df['bhk'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define range of k
k_range = range(1, 12)

# Get accuracy values
acc_list = accuracy_for_k_range(X_train, X_test, y_train, y_test, k_range)

# Print accuracy results
print("=== A8: Accuracy for different k values ===")
for k, acc in zip(k_range, acc_list):
    print(f"k = {k}, Accuracy = {acc:.4f}")

# Plotting accuracy vs k
plt.plot(k_range, acc_list, marker='o')
plt.xlabel("k (Number of Neighbors)")
plt.ylabel("Accuracy")
plt.title("k-NN Accuracy for k = 1 to 11")
plt.grid(True)
plt.show()
