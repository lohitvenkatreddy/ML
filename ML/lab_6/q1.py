'''Write a function to calculate the entropy associated with your dataset. (If your dataset contains
only continuous numeric data for outcome (a regression problem), employ equal width binning and
divide your data into 4 bins. Each bin may be considered as a categorical data value. Write a function
for equal width binning).'''
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("top_2_hdd_models.csv")
'''
# Equal width binning function
def equal_width_binning(series, n_bins=4):
    bins = np.linspace(series.min(), series.max(), n_bins+1)
    return pd.cut(series, bins=bins, labels=False, include_lowest=True)
'''
# Entropy calculation
def entropy(y):
    values, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    return -np.sum(probabilities * np.log2(probabilities))

if __name__ == "__main__":
    # Example: assume 'failure' column is the target
    target = df['failure']
    # If numeric → bin
    #if pd.api.types.is_numeric_dtype(target):
        #target = equal_width_binning(target, 4)
    calculate_entropy = entropy(target)
    print("Entropy of dataset:", calculate_entropy)
