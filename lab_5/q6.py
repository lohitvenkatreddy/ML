import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def preprocess_features(df):
    print(f"Initial shape: {df.shape}")
    
    df = df.select_dtypes(include=[np.number])
    print(f"After keeping only numeric columns: {df.shape}")
    
    imputer = SimpleImputer(strategy='mean')
    df_imputed = imputer.fit_transform(df)
    print(f"After imputation: {df_imputed.shape}")
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_imputed)
    print("Data scaled.")
    
    return scaled_data

def evaluate_kmeans(X, k_range):
    s_scores, ch_scores, db_scores = [], [], []

    for k in k_range:
        print(f"Clustering for k={k}...")
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(X)

        s = silhouette_score(X, labels)
        ch = calinski_harabasz_score(X, labels)
        db = davies_bouldin_score(X, labels)

        s_scores.append(s)
        ch_scores.append(ch)
        db_scores.append(db)

        print(f"k={k}: Silhouette={s:.4f}, Calinski-Harabasz={ch:.2f}, Davies-Bouldin={db:.4f}")

    return s_scores, ch_scores, db_scores

def plot_scores(k_range, s_scores, ch_scores, db_scores):
    best_s_index = np.argmax(s_scores)
    best_ch_index = np.argmax(ch_scores)
    best_db_index = np.argmin(db_scores)

    plt.figure(figsize=(16, 5))

    plt.subplot(1, 3, 1)
    plt.plot(k_range, s_scores, marker='o')
    plt.title('Silhouette Score')
    plt.xlabel('k')
    plt.ylabel('Score')
    plt.axvline(k_range[best_s_index], color='r', linestyle='--', label=f'Optimal k={k_range[best_s_index]}')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(k_range, ch_scores, marker='o')
    plt.title('Calinski-Harabasz Score')
    plt.xlabel('k')
    plt.ylabel('Score')
    plt.axvline(k_range[best_ch_index], color='g', linestyle='--', label=f'Optimal k={k_range[best_ch_index]}')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(k_range, db_scores, marker='o')
    plt.title('Davies-Bouldin Score')
    plt.xlabel('k')
    plt.ylabel('Score (lower is better)')
    plt.axvline(k_range[best_db_index], color='orange', linestyle='--', label=f'Optimal k={k_range[best_db_index]}')
    plt.legend()

    plt.tight_layout()
    plt.show()

# === Load dataset ===
file_path = "top_2_hdd_models.csv"
df = pd.read_csv(file_path)

# === Sample for performance ===
df_sampled = df.sample(n=10000, random_state=42)

# === Preprocess ===
X = preprocess_features(df_sampled)

# === Evaluate k-means ===
k_values = range(2, 11)
silhouette_scores, ch_scores, db_scores = evaluate_kmeans(X, k_values)

# === Plot with optimal k marked ===
plot_scores(k_values, silhouette_scores, ch_scores, db_scores)
