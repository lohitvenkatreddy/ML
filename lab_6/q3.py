import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

df = pd.read_csv("top_2_hdd_models.csv")

def entropy(y):
    values, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    return -np.sum(probabilities * np.log2(probabilities))

def information_gain(y, x):
    base_entropy = entropy(y)
    values, counts = np.unique(x, return_counts=True)
    weighted_entropy = 0
    for v, c in zip(values, counts):
        subset_y = y[x == v]
        weighted_entropy += (c/len(x)) * entropy(subset_y)
    return base_entropy - weighted_entropy

def preprocess_feature(series, n_bins=4, strategy="uniform"):
    """Convert numeric feature into categorical bins if needed, handling NaN & empty cols"""
    if series.isna().all():
        # Entire column is NaN → drop it
        return None  
    
    if pd.api.types.is_numeric_dtype(series):
        series = series.fillna(series.median())  # fill numeric NaN with median
        if series.nunique() <= 1:
            # Constant feature → useless
            return None
        est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
        return est.fit_transform(series.values.reshape(-1,1)).astype(int).ravel()
    else:
        series = series.fillna(series.mode()[0])  # fill categorical NaN with mode
        return series

def choose_root_node(X, y, n_bins=4, strategy="uniform"):
    gains = {}
    for col in X.columns:
        x_processed = preprocess_feature(X[col], n_bins=n_bins, strategy=strategy)
        if x_processed is None:  # skip empty or constant columns
            continue
        gains[col] = information_gain(y, x_processed)
    return max(gains, key=gains.get), gains

if __name__ == "__main__":
    X = df.drop(columns=['failure'])
    y = df['failure'].fillna(df['failure'].mode()[0])  # ensure target has no NaN
    
    root, gains = choose_root_node(X, y, n_bins=4, strategy="uniform")  # uniform = equal-width
    print("Best root node:", root)
    print("Top 5 features by Information Gain:")
    for feature, val in sorted(gains.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {feature}: {val:.4f}")
