'''. Please refer to the “Purchase Data” worksheet of Lab Session Data.xlsx. Please load the data
and segregate them into 2 matrices A & C (following the nomenclature of AX = C). Do the following
activities.
• What is the dimensionality of the vector space for this data?
• How many vectors exist in this vector space?
• What is the rank of Matrix A?
• Using Pseudo-Inverse find the cost of each product available for sale.
(Suggestion: If you use Python, you can use numpy.linalg.pinv() function to get a
pseudo-inverse.)'''



import pandas as pd
import numpy as np

def load_purchase_data(file_path, sheet_name):
    return pd.read_excel(file_path, sheet_name=sheet_name)

def get_matrices(data, A_columns, C_column):
    A = data[A_columns].values
    C = data[C_column].values.reshape(-1, 1)
    return A, C

def get_dimensionality(A):
    return A.shape[1]

def get_num_vectors(A):
    return A.shape[0]

def get_rank(A):
    return np.linalg.matrix_rank(A)

def compute_product_costs(A, C):
    A_pinv = np.linalg.pinv(A)
    X = A_pinv @ C
    return X

if __name__ == "__main__":
    file_path = "LabData.xlsx"
    sheet_name = "Purchase data"
    A_columns = ['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']
    C_column = 'Payment (Rs)'

    data = load_purchase_data(file_path, sheet_name)
    A, C = get_matrices(data, A_columns, C_column)
    dimensionality = get_dimensionality(A)
    num_vectors = get_num_vectors(A)
    rank_A = get_rank(A)
    product_costs = compute_product_costs(A, C)

    print(f"Dimensionality of vector space: {dimensionality}")
    print(f"Number of vectors in vector space: {num_vectors}")
    print(f"Rank of Matrix A: {rank_A}")
    print("\nCost of each product:")
    for i, cost in enumerate(product_costs, start=1):
        print(f"Product {i}: Rs. {cost[0]:.2f}")
