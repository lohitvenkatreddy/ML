import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def detect_outliers(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return ((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).any()

def impute_and_normalize(df):
    df = df.replace("?", np.nan)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            if detect_outliers(df[col].dropna()):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mean())

    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df, numeric_cols

def main():
    df = pd.read_excel("LabData.xlsx", sheet_name="thyroid0387_UCI")
    df.replace("?", np.nan, inplace=True)
    normalized_df, cols = impute_and_normalize(df)
    print("A9 Result (Fixed):")
    print("Normalized numeric attributes:", cols)
    print(normalized_df[cols].head())

main()