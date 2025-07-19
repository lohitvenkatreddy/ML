import pandas as pd
import numpy as np
from scipy.stats import mode

def detect_outliers(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return ((series < lower) | (series > upper)).any()

def impute_data(df):
    df = df.replace("?", np.nan)
    for col in df.columns:
        if df[col].dtype == object:
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            if df[col].isnull().sum() > 0:
                if detect_outliers(df[col].dropna()):
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna(df[col].mean(), inplace=True)
    return df

def main():
    df = pd.read_excel("LabData.xlsx", sheet_name="thyroid0387_UCI")
    df.replace("?", np.nan, inplace=True)
    df_imputed = impute_data(df)
    print("A8 Result:")
    print("Missing values after imputation:\n", df_imputed.isnull().sum())

main()