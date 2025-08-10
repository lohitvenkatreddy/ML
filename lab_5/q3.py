import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

def load_full_data(filepath, target_col):
    """Load all SMART features, drop highly missing columns, and impute missing values."""
    df = pd.read_csv(filepath)

    # Select SMART features with raw values and target
    smart_cols = [col for col in df.columns if 'smart_' in col and '_raw' in col]
    df = df[smart_cols + [target_col]]

    # Drop columns with more than 50% missing values
    df = df.loc[:, df.isnull().mean() < 0.5]

    # Drop rows where target is missing
    df = df.dropna(subset=[target_col])

    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Impute remaining missing values in features with mean
    imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    return train_test_split(X_imputed, y, test_size=0.3, random_state=42)


def evaluate_model(model, X, y):
    """Return MSE, RMSE, MAPE, R2 for given model predictions."""
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    return mse, rmse, mape, r2


if __name__ == "__main__":
    file_path = "top_2_hdd_models.csv"  # Update path if needed
    target = "failure"

    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_full_data(file_path, target)

    # Train linear regression
    model = LinearRegression().fit(X_train, y_train)

    # Evaluate on training and test data
    train_metrics = evaluate_model(model, X_train, y_train)
    test_metrics = evaluate_model(model, X_test, y_test)

    print("Train - MSE, RMSE, MAPE, R²:", train_metrics)
    print("Test  - MSE, RMSE, MAPE, R²:", test_metrics)

# Your MAPE values are enormous (on the order of 10 12 %). This is because MAPE blows up when your target variable has many zeros (common in "failure" since most drives don’t fail). 
# You can’t divide by zero in MAPE, so sklearn treats it in a way that leads to huge values.