'''. Calculate MSE, RMSE, MAPE and R2 scores for the price prediction exercise done in Lab 02.
Analyse the results.'''
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

def load_file(filepath, sheet_name):
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    df.columns = df.columns.str.strip()
    return df

def matrices(data, Acolumns, Ccolumn):
    A = data[Acolumns].values
    C = data[Ccolumn].values.reshape(-1, 1)
    return A, C

def dimensionality(A):
    return A.shape[1]

def vectors(A):
    return A.shape[0]

def rank(A):
    return np.linalg.matrix_rank(A)

def cost(A, C):
    A_pinv = np.linalg.pinv(A)
    X = A_pinv @ C
    return X

def predicted_C(A, cost_vector):
    return A @ cost_vector

def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mape, r2

if __name__ == "__main__":
    filepath = "LabData.xlsx"
    sheetname = "Purchase data"
    Acolumn = ['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']
    Ccolumn = 'Payment (Rs)'

    data = load_file(filepath, sheetname)
    A, C = matrices(data, Acolumn, Ccolumn)

    dim = dimensionality(A)
    numvectors = vectors(A)
    rank_A = rank(A)
    X = cost(A, C)
    C_pred = predicted_C(A, X)

    mse, rmse, mape, r2 = regression_metrics(C, C_pred)

    print(f"Dimensionality of vector space: {dim}")
    print(f"Number of vectors in vector space: {numvectors}")
    print(f"Rank of Matrix A: {rank_A}")
    print("\nCost of each product:")
    for i, cost in enumerate(X, start=1):
        print(f"Product {i}: Rs. {cost[0]:.2f}")

    print("\nRegression Metrics (A2):")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape*100:.2f}%")
    print(f"R-squared Score (R²): {r2:.4f}")