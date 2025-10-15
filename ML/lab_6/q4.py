''' If the feature is continuous valued for A3, use equal width or frequency binning for converting
the attribute to categorical valued. The binning type should be a parameter to the function built for
binning. Write your own function for the binning task. The number of bins to be created should also
be passed as a parameter to the function. Use function overloading to allow for usage of default
parameters if no parameters are passed.'''
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
    binning = equal_width_binning(df[col], 4)[:10]
    print("Equal width binning:\n", equal_width_binning(df[col], 4)[:10])
    print("Frequency binning:\n", frequency_binning(df[col], 4)[:10])
