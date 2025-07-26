import pandas as pd
import numpy as np

def compute_class_stats(df, class_column, class_values, features):
    df_filtered = df[df[class_column].isin(class_values)]
    class1_data = df_filtered[df_filtered[class_column] == class_values[0]][features].values
    class2_data = df_filtered[df_filtered[class_column] == class_values[1]][features].values

    class1_mean = np.mean(class1_data, axis=0)
    class2_mean = np.mean(class2_data, axis=0)

    class1_std = np.std(class1_data, axis=0)
    class2_std = np.std(class2_data, axis=0)

    interclass_distance = np.linalg.norm(class1_mean - class2_mean)

    return class1_mean, class2_mean, class1_std, class2_std, interclass_distance

# Load dataset
df = pd.read_csv("Hyderbad_House_price.csv")
df['bhk'] = df['title'].str.extract(r'(\d) BHK').astype(float)
df_bhk = df[df['bhk'].isin([2, 3])]

features = ['price(L)', 'rate_persqft', 'area_insqft']
class1_mean, class2_mean, class1_std, class2_std, inter_dist = compute_class_stats(df_bhk, 'bhk', (2, 3), features)
# A1: Print outputs
print("2 BHK Mean Vector:", class1_mean)
print("3 BHK Mean Vector:", class2_mean)
print("2 BHK Std Dev:", class1_std)
print("3 BHK Std Dev:", class2_std)
print("Interclass Distance:", inter_dist)
