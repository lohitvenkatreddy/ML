'''Cosine Similarity Measure: Now take the complete vectors for these two observations (including
all the attributes). Calculate the Cosine similarity between the documents by using the second
feature vector for each document.'''
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

def compute_cosine_similarity(df):
    df = df.copy()
    df = df.replace({'f': 0, 't': 1, 'n': 0, 'y': 1, '?': np.nan})
    df = df.fillna(0)

    for col in df.columns:
        if df[col].dtype == object:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    vec1 = df.iloc[0].values.reshape(1, -1)
    vec2 = df.iloc[1].values.reshape(1, -1)
    sim = cosine_similarity(vec1, vec2)[0][0]
    return sim

def main():
    df = pd.read_excel("LabData.xlsx", sheet_name="thyroid0387_UCI")
    similarity = compute_cosine_similarity(df)
    print("A6 Result:")
    print("Cosine Similarity between first two full vectors:", similarity)

main()