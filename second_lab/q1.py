import pandas as pd
import numpy as np

df = pd.read_excel("labdata.xlsx", sheet_name=0)

# Drop unnamed or empty columns
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Automatically detect numeric columns between first and last
product_columns = df.columns[1:-1]
target_column = df.columns[-1]

A_df = df[product_columns]
C_series = df[target_column]

A_df = A_df.apply(pd.to_numeric, errors='coerce')
C_series = pd.to_numeric(C_series, errors='coerce')

A = A_df.fillna(0).values
C = C_series.fillna(0).values

dimensionality = A.shape[1]
num_vectors = A.shape[0]
rank = np.linalg.matrix_rank(A)

A_pinv = np.linalg.pinv(A)
X = A_pinv @ C

cost_per_product = pd.Series(X, index=product_columns)

print("Dimensionality of the vector space:", dimensionality)
print("Number of vectors in the vector space:", num_vectors)
print("Rank of Matrix A:", rank)
print("Estimated cost per product:")
print(cost_per_product)
