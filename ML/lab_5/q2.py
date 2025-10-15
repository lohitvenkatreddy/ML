'''Calculate MSE, RMSE, MAPE and R2 scores for prediction made by the trained model in A1.
Perform prediction on the test data and compare the metric values between train and test set.'''
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def load_and_prepare_data(filepath, feature_col, target_col):
    df = pd.read_csv(filepath)
    df = df[[feature_col, target_col]].dropna()
    X = df[[feature_col]]
    y = df[target_col]
    return train_test_split(X, y, test_size=0.3, random_state=42)

def safe_mape(y_true, y_pred):
    # Avoid division by zero issues for classification targets
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = y_true != 0
    if not np.any(non_zero_mask):
        return np.nan
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mape = safe_mape(y, y_pred)
    r2 = r2_score(y, y_pred)
    return mse, rmse, mape, r2

if __name__ == "__main__":
    file_path = "top_2_hdd_models.csv"
    feature = "smart_1_normalized"  # same as A1
    target = "failure"  # classification target

    X_train, X_test, y_train, y_test = load_and_prepare_data(file_path, feature, target)
    model = LinearRegression().fit(X_train, y_train)

    train_metrics = evaluate_model(model, X_train, y_train)
    test_metrics = evaluate_model(model, X_test, y_test)

    print("Train - MSE, RMSE, MAPE (%), R²:", train_metrics)
    print("Test  - MSE, RMSE, MAPE (%), R²:", test_metrics)
#MSE / RMSE → “How far are we off, on average?” (distance-based error)

#MAPE → “How wrong are we in percentage terms?”

#R² → “How much of the pattern did we actually capture?”