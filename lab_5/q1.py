import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Function to load and clean data
def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    # Remove non-numeric columns
    non_numeric_cols = df.select_dtypes(include=['object']).columns
    df = df.drop(columns=non_numeric_cols)
    # Remove columns with >50% missing values
    df = df.dropna(thresh=int(0.5 * df.shape[0]), axis=1)
    return df

# Function to preprocess features using only one attribute as per A1
def preprocess_one_feature(df, feature_col, target_col):
    X = df[[feature_col]]  # Only one feature
    y = df[target_col]     # Target variable
    
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    # Scale the feature
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    return X_scaled, y

# Function to split data
def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Function to train linear regression model
def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# MAIN PROGRAM
filepath = "top_2_hdd_models.csv"
target_col = "failure"           # Replace with your actual target column
feature_col = "smart_1_normalized"  # Example numerical attribute — change to your dataset's

df = load_and_clean_data(filepath)
X, y = preprocess_one_feature(df, feature_col, target_col)
X_train, X_test, y_train, y_test = split_data(X, y)

model = train_linear_regression(X_train, y_train)

# Output as per instructions
print("Model Coefficient:", model.coef_)
print("Model Intercept:", model.intercept_)
