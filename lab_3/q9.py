import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Function to evaluate model performance
def evaluate_model_performance(model, X_data, y_data):
    predictions = model.predict(X_data)
    cm = confusion_matrix(y_data, predictions)
    report = classification_report(y_data, predictions, output_dict=False)
    return cm, report

# Load and preprocess dataset
df = pd.read_csv("Hyderbad_House_price.csv")
df['bhk'] = df['title'].str.extract(r'(\d) BHK').astype(float)
df = df[df['bhk'].isin([2, 3])]  # Binary classification

# Select features and labels
features = ['price(L)', 'rate_persqft', 'area_insqft']
X = df[features].values
y = df['bhk'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train kNN model
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

# Evaluate on training data
cm_train, report_train = evaluate_model_performance(knn_model, X_train, y_train)

# Evaluate on test data
cm_test, report_test = evaluate_model_performance(knn_model, X_test, y_test)

# Print evaluation results
print("=== A9: Model Evaluation ===")
print("\nTraining Confusion Matrix:\n", cm_train)
print("\nTraining Classification Report:\n", report_train)

print("\nTest Confusion Matrix:\n", cm_test)
print("\nTest Classification Report:\n", report_test)

# Determine learning outcome
print("\nLearning Outcome Analysis:")
train_accuracy = knn_model.score(X_train, y_train)
test_accuracy = knn_model.score(X_test, y_test)
print(f"Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")

if train_accuracy > 0.95 and test_accuracy < 0.80:
    print("Model may be overfitting.")
elif abs(train_accuracy - test_accuracy) < 0.05:
    print("Model is likely regularfit (generalizing well).")
else:
    print("Model may be underfitting or inconsistent.")
