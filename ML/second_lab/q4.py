'''Data Exploration: Load the data available in “thyroid0387_UCI” worksheet. Perform the
following tasks:
• Study each attribute and associated values present. Identify the datatype (nominal etc.)
for the attribute.
• For categorical attributes, identify the encoding scheme to be employed. (Guidance:
employ label encoding for ordinal variables while One-Hot encoding may be employed
for nominal variables).
• Study the data range for numeric variables.
• Study the presence of missing values in each attribute.
• Study presence of outliers in data.
• For numeric variables, calculate the mean and variance (or standard deviation).'''
import pandas as pd
import numpy as np

def analyze_thyroid_data(df):
    info = {}

    types = df.dtypes
    nominal = []
    ordinal = []
    numeric = []

    for col in df.columns:
        if df[col].dtype == 'object':
            unique_vals = df[col].dropna().unique()
            if col in ['on thyroxine', 'query on thyroxine', 'on antithyroid medication',
                       'sick', 'pregnant', 'thyroid surgery', 'I131 treatment', 
                       'query hypothyroid', 'query hyperthyroid', 'lithium', 
                       'goitre', 'tumor', 'hypopituitary', 'psych']:
                ordinal.append(col)
            else:
                nominal.append(col)
        else:
            numeric.append(col)

    info['nominal'] = nominal
    info['ordinal'] = ordinal
    info['numeric'] = numeric

    encoding = {}
    for col in ordinal:
        encoding[col] = "Label Encoding"
    for col in nominal:
        encoding[col] = "One-Hot Encoding"
    info['encoding'] = encoding

    ranges = {}
    for col in numeric:
        ranges[col] = (float(df[col].min()), float(df[col].max()))
    info['ranges'] = ranges

    info['missing'] = df.isnull().sum()

    outliers = {}
    for col in numeric:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers[col] = df[(df[col] < lower) | (df[col] > upper)].shape[0]
    info['outliers'] = outliers

    means = {}
    stds = {}
    for col in numeric:
        means[col] = float(df[col].mean())
        stds[col] = float(df[col].std())
    info['mean'] = means
    info['std'] = stds

    return info

def main():
    df = pd.read_excel("LabData.xlsx", sheet_name="thyroid0387_UCI")
    info = analyze_thyroid_data(df)

    print("A4 Results:")
    print("\nNominal Attributes:", info['nominal'])
    print("\nOrdinal Attributes:", info['ordinal'])
    print("\nNumeric Attributes:", info['numeric'])

    print("\nEncoding Recommendation:")
    for col, enc in info['encoding'].items():
        print(f"  {col}: {enc}")

    print("\nNumeric Ranges:")
    for col, (min_val, max_val) in info['ranges'].items():
        print(f"  {col}: ({min_val:.2f}, {max_val:.2f})")

    print("\nMissing Values:")
    print(info['missing'])

    print("\nOutliers (count):")
    for col, count in info['outliers'].items():
        print(f"  {col}: {count}")

    print("\nMeans:")
    for col, val in info['mean'].items():
        print(f"  {col}: {val:.2f}")

    print("\nStandard Deviations:")
    for col, val in info['std'].items():
        print(f"  {col}: {val:.2f}")

main()