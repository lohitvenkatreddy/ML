import pandas as pd
import numpy as np

df = pd.read_csv("top_2_hdd_models.csv")

def gini_index(y):
    values, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    return 1 - np.sum(probabilities ** 2)

if __name__ == "__main__":
    target = df['failure']
    print("Gini Index of dataset:", gini_index(target))
