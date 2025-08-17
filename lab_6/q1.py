import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("top_2_hdd_models.csv")

# Equal width binning function
def equal_width_binning(series, n_bins=4):
    bins = np.linspace(series.min(), series.max(), n_bins+1)
    return pd.cut(series, bins=bins, labels=False, include_lowest=True)

# Entropy calculation
def entropy(y):
    values, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    return -np.sum(probabilities * np.log2(probabilities))

if __name__ == "__main__":
    # Example: assume 'failure' column is the target
    target = df['failure']  
    # If numeric → bin
    if pd.api.types.is_numeric_dtype(target):
        target = equal_width_binning(target, 4)
    print("Entropy of dataset:", entropy(target))
