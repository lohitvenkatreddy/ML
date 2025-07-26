import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Function to evaluate accuracy
def evaluate_model_accuracy(model, X_test, y_test):
    return model.score(X_test, y_test)

# Load dataset and preprocess
df = pd.read_csv("Hyderbad_House_price.csv")
df['bhk'] = df['title'].str.extract(r'(\d) BHK').astype(float)
df = df[df['bhk'].isin([2, 3])]

# Feature and label extraction
features = ['price(L)', 'rate_persqft', 'area_insqft']
X = df[features].values
y = df['bhk'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the kNN model
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

# Evaluate accuracy
accuracy = evaluate_model_accuracy(knn_model, X_test, y_test)

# Print result
print("=== A6: Model Accuracy (k=3) ===")
print("Accuracy on test set:", round(accuracy * 100, 2), "%")
