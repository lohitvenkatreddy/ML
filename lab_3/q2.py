import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def analyze_feature_distribution(data, feature_name):
    values = data[feature_name].values
    mean_val = np.mean(values)
    var_val = np.var(values)

    plt.hist(values, bins=20, edgecolor='black')
    plt.title(f"Histogram of {feature_name}")
    plt.xlabel(feature_name)
    plt.ylabel("Frequency")
    plt.show()

    return mean_val, var_val

# Load dataset
df = pd.read_csv("Hyderbad_House_price.csv")
df['bhk'] = df['title'].str.extract(r'(\d) BHK').astype(float)
df_bhk = df[df['bhk'].isin([2, 3])]

mean_feat, var_feat = analyze_feature_distribution(df, 'rate_persqft')

print("=== A2: Histogram Analysis ===")
print("Mean of 'rate_persqft':", mean_feat)
print("Variance of 'rate_persqft':", var_feat)
