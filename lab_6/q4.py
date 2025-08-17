import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

df = pd.read_csv("top_2_hdd_models.csv")

def equal_width_binning(series, n_bins=4):
    bins = np.linspace(series.min(), series.max(), n_bins+1)
    return pd.cut(series, bins=bins, labels=False, include_lowest=True)

def frequency_binning(series, n_bins=4):
    est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    return est.fit_transform(series.to_numpy().reshape(-1, 1)).astype(int).ravel()

if __name__ == "__main__":
    col = df.select_dtypes(include=[np.number]).columns[0]
    print("Equal width binning:\n", equal_width_binning(df[col], 4)[:10])
    print("Frequency binning:\n", frequency_binning(df[col], 4)[:10])
