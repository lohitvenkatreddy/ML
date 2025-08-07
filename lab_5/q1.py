import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def load_and_prepare_data(filepath, feature_col, target_col):
    df = pd.read_csv(filepath)
    df = df[[feature_col, target_col]].dropna()
    X = df[[feature_col]]
    y = df[target_col]
    return train_test_split(X, y, test_size=0.3, random_state=42)

def train_single_feature_regression(X_train, y_train):
    model = LinearRegression().fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    return model, y_train_pred

if __name__ == "__main__":
    file_path = "top_2_hdd_models.csv"
    feature = "smart_5_raw"
    target = "failure"

    X_train, X_test, y_train, y_test = load_and_prepare_data(file_path, feature, target)
    model, y_train_pred = train_single_feature_regression(X_train, y_train)

    print("Model Coefficient:", model.coef_)
    print("Model Intercept:", model.intercept_)
