import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Function to train kNN
def train_knn_classifier(X_train, y_train, k=3):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    return model

# Load and filter data
df = pd.read_csv("Hyderbad_House_price.csv")
df['bhk'] = df['title'].str.extract(r'(\d) BHK').astype(float)
df = df[df['bhk'].isin([2, 3])]

# Define features
features = ['price(L)', 'rate_persqft', 'area_insqft']
X = df[features].values
y = df['bhk'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
knn_model = train_knn_classifier(X_train, y_train, k=3)

# Print model training complete
print("=== A5: kNN Classifier Trained (k=3) ===")
print("Training samples:", len(X_train))
print("Test samples:", len(X_test))
