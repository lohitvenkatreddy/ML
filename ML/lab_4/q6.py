'''Repeat the exercises A3 to A5 for your project data
 considering any two features and classes.'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

def load_and_balance_data(filepath, feature_cols, label_col, samples_per_class=200):
    """
    Load the dataset and balance classes for binary classification
    """
    df = pd.read_csv(filepath)
    df = df.dropna(subset=feature_cols + [label_col])

    # Convert labels to numeric if necessary
    if df[label_col].dtype == 'object':
        encoder = LabelEncoder()
        df[label_col] = encoder.fit_transform(df[label_col])

    # Separate classes
    class_0 = df[df[label_col] == 0]
    class_1 = df[df[label_col] == 1]

    # Resample to balance
    class_0_sampled = resample(class_0, n_samples=samples_per_class, random_state=42)
    class_1_sampled = resample(class_1, n_samples=samples_per_class, random_state=42)

    df_balanced = pd.concat([class_0_sampled, class_1_sampled])
    X = df_balanced[feature_cols].values
    y = df_balanced[label_col].values

    return X, y

def train_knn_classifier(X_train, y_train, k=3):
    model = KNeighborsClassifier(n_neighbors=k, algorithm='kd_tree')
    model.fit(X_train, y_train)
    return model

def visualize_classification(X, y_pred, feature_names, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='bwr', s=60, edgecolor='k')
    plt.title(title)
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ========== MAIN ==========
filepath = "top_2_hdd_models.csv"
features = ['smart_5_raw', 'smart_187_raw']
label = 'failure'

# Load and balance the data
X_balanced, y_balanced = load_and_balance_data(filepath, features, label, samples_per_class=200)

# Train model and predict
knn_model = train_knn_classifier(X_balanced, y_balanced, k=3)
y_pred = knn_model.predict(X_balanced)

# Visualize
visualize_classification(X_balanced, y_pred, features, "A6: kNN Classification on Balanced Project Data (k=3)")
