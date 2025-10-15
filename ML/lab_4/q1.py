'''Please evaluate confusion matrix for your classification problem. From confusion matrix, the
other performance metrics such as precision, recall and F1-Score measures for both training and test
data. Based on your observations, infer the models learning outcome (underfit / regularfit / overfit). '''
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    print("Initial size:", df.shape[0])
    
    non_numeric_cols = df.select_dtypes(include=['object']).columns
    df = df.drop(columns=non_numeric_cols)
    
    df = df.dropna(thresh=int(0.5 * df.shape[0]), axis=1)
    
    return df

def preprocess_features(df, target_col="failure"):
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    return X_scaled, y

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_knn_model(X_train, y_train, n_neighbors=3):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X, y, dataset_name=""):
    y_pred = model.predict(X)
    print(f"\n{dataset_name} Confusion Matrix:\n", confusion_matrix(y, y_pred))
    print(f"\n{dataset_name} Classification Report:\n", classification_report(y, y_pred))


filepath = "top_2_hdd_models.csv"
    
df = load_and_clean_data(filepath)
X, y = preprocess_features(df)
X_train, X_test, y_train, y_test = split_data(X, y)
    
knn_model = train_knn_model(X_train, y_train)
    
evaluate_model(knn_model, X_train, y_train, "Train")
evaluate_model(knn_model, X_test, y_test, "Test")
