import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score

def load_clustering_data(filepath, target_col):
    """Load SMART data, drop columns with >50% NaNs, keep target separately."""
    df = pd.read_csv(filepath)
    smart_cols = [col for col in df.columns if 'smart_' in col and '_raw' in col]
    df = df[smart_cols + [target_col]]
    df = df.loc[:, df.isnull().mean() < 0.5].dropna()
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def perform_kmeans(X, k=2):
    """Perform KMeans clustering."""
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    return labels, kmeans.cluster_centers_

if __name__ == "__main__":
    file_path = "top_2_hdd_models.csv"
    target = "failure"

    # Load data
    X, y = load_clustering_data(file_path, target)

    # Dimensionality reduction for speed
    pca = PCA(n_components=10, random_state=42)
    X_pca = pca.fit_transform(X)

    # Clustering
    labels, centers = perform_kmeans(X_pca, k=2)

    print("First 10 Cluster Labels:", labels[:10])
    print("Cluster Centers Shape:", centers.shape)

    # Compare clusters with actual labels
    cm = confusion_matrix(y, labels)
    acc = accuracy_score(y, labels)

    print("\nConfusion Matrix (rows: actual, cols: clusters):\n", cm)
    print("Clustering Accuracy:", acc)
