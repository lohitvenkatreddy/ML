import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Function to predict labels
def predict_test_samples(model, test_features):
    return model.predict(test_features)

# Load and preprocess data
df = pd.read_csv("Hyderbad_House_price.csv")
df['bhk'] = df['title'].str.extract(r'(\d) BHK').astype(float)
df = df[df['bhk'].isin([2, 3])]  # Take only 2 BHK and 3 BHK for binary classification

# Feature and label selection
features = ['price(L)', 'rate_persqft', 'area_insqft']
X = df[features].values
y = df['bhk'].values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train kNN model
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

# Predict test labels
predicted_labels = predict_test_samples(knn_model, X_test)

# Predict a single sample (optional: first row)
single_prediction = predict_test_samples(knn_model, X_test[:1])

# Print results
print("=== A7: kNN Prediction ===")
print("Predicted Labels for Test Set:", predicted_labels[:10], "...")
print("Actual Labels for Test Set:   ", y_test[:10], "...")
print("Prediction for 1st Test Sample:", single_prediction[0])
