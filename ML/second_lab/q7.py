'''Heatmap Plot: Consider the first 20 observation vectors. Calculate the JC, SMC and COS between
the pairs of vectors for these 20 vectors. Employ similar strategies for coefficient calculation as in A4
& A5. Employ a heatmap plot to visualize the similarities. '''
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

def preprocess(df):
    df = df.replace({'f': 0, 't': 1, 'n': 0, 'y': 1, '?': np.nan})
    df = df.fillna(0)
    for col in df.columns:
        if df[col].dtype == object:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    return df

def calculate_jaccard(v1, v2):
    f11 = np.sum((v1 == 1) & (v2 == 1))
    f10 = np.sum((v1 == 1) & (v2 == 0))
    f01 = np.sum((v1 == 0) & (v2 == 1))
    denom = f11 + f10 + f01
    return f11 / denom if denom != 0 else 0

def calculate_smc(v1, v2):
    f11 = np.sum((v1 == 1) & (v2 == 1))
    f00 = np.sum((v1 == 0) & (v2 == 0))
    f10 = np.sum((v1 == 1) & (v2 == 0))
    f01 = np.sum((v1 == 0) & (v2 == 1))
    total = f11 + f00 + f10 + f01
    return (f11 + f00) / total if total != 0 else 0

def compute_similarity_matrices(df):
    df = preprocess(df)
    df_20 = df.iloc[:20]

    jc_matrix = np.zeros((20, 20))
    smc_matrix = np.zeros((20, 20))
    cos_matrix = np.zeros((20, 20))

    for i in range(20):
        for j in range(20):
            v1 = df_20.iloc[i].values
            v2 = df_20.iloc[j].values
            jc_matrix[i, j] = calculate_jaccard(v1, v2)
            smc_matrix[i, j] = calculate_smc(v1, v2)
            cos_matrix[i, j] = cosine_similarity([v1], [v2])[0][0]

    return jc_matrix, smc_matrix, cos_matrix

def plot_heatmap(data, title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(data, annot=True, cmap="viridis", fmt=".2f")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def main():
    df = pd.read_excel("LabData.xlsx", sheet_name="thyroid0387_UCI")
    jc_matrix, smc_matrix, cos_matrix = compute_similarity_matrices(df)
    print("A7 Results: Similarity matrices generated for first 20 records.")
    plot_heatmap(jc_matrix, "Jaccard Coefficient Heatmap (First 20)")
    plot_heatmap(smc_matrix, "Simple Matching Coefficient Heatmap (First 20)")
    plot_heatmap(cos_matrix, "Cosine Similarity Heatmap (First 20)")

main()