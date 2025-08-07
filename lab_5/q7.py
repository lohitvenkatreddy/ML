import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans

# === Load Dataset ===
file_path = "top_2_hdd_models.csv"  # Adjust as needed
df = pd.read_csv(file_path)

# === Extract numeric columns ===
df_numeric = df.select_dtypes(include=[np.number])

# === Drop columns with all missing values ===
df_numeric = df_numeric.dropna(axis=1, how='all')

# === Sample 10,000 rows for faster elbow detection ===
df_sampled = df_numeric.sample(n=10000, random_state=42)

# === Impute missing values (mean) ===
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df_sampled), columns=df_sampled.columns)

# === Standardize features ===
scaler = StandardScaler()
X_train = scaler.fit_transform(df_imputed)

# === Elbow Method ===
distortions = []
k_range = range(2, 20)

for k in k_range:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(X_train)
    distortions.append(kmeans.inertia_)

# === Plot Elbow Curve ===
plt.figure(figsize=(8, 5))
plt.plot(k_range, distortions, marker='o')
plt.title('Elbow Method for Optimal k (Sampled 10,000 Rows)')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (WCSS)')
plt.xticks(k_range)
plt.grid(True)
plt.tight_layout()
plt.show()
