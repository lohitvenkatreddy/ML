import pandas as pd
from sklearn.cluster import KMeans

def load_clustering_data(filepath, target_col):
    """Load data, drop high-NaN columns, and remove the target column."""
    df = pd.read_csv(filepath)
    
    # Select raw SMART attributes
    smart_cols = [col for col in df.columns if 'smart_' in col and '_raw' in col]
    df = df[smart_cols + [target_col]]
    
    # Drop columns with >50% missing
    df = df.loc[:, df.isnull().mean() < 0.5]
    
    # Drop rows with missing values
    df = df.dropna()

    # Drop the target variable
    X = df.drop(columns=[target_col])

    return X

def perform_kmeans(X, k=2):
    """Fit KMeans clustering and return labels and cluster centers."""
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X)
    return kmeans.labels_, kmeans.cluster_centers_

if __name__ == "__main__":
    file_path = "top_2_hdd_models.csv"
    target = "failure"

    X = load_clustering_data(file_path, target)
    labels, centers = perform_kmeans(X, k=2)

    print("First 10 Cluster Labels:", labels[:10])
    print("\nCluster Centers Shape:", centers.shape)
