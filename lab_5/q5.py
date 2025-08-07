import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def load_clustering_data(filepath, target_col):
    """Load SMART data for clustering, drop high-NaN columns and target column."""
    df = pd.read_csv(filepath)
    
    smart_cols = [col for col in df.columns if 'smart_' in col and '_raw' in col]
    df = df[smart_cols + [target_col]]
    
    # Drop columns with too many missing values
    df = df.loc[:, df.isnull().mean() < 0.5]

    # Drop rows with missing data
    df = df.dropna()

    # Remove target column (unsupervised)
    X = df.drop(columns=[target_col])
    
    return X

def evaluate_kmeans_clustering(X, k=2):
    """Fit KMeans and return silhouette, CH, and DB scores."""
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X)
    labels = kmeans.labels_

    # Calculate clustering metrics
    sil_score = silhouette_score(X, labels)
    ch_score = calinski_harabasz_score(X, labels)
    db_score = davies_bouldin_score(X, labels)

    return sil_score, ch_score, db_score

if __name__ == "__main__":
    file_path = "top_2_hdd_models.csv"
    target = "failure"

    X = load_clustering_data(file_path, target)
    sil, ch, db = evaluate_kmeans_clustering(X, k=2)

    print("Silhouette Score:", sil)
    print("Calinski-Harabasz Score:", ch)
    print("Davies-Bouldin Index:", db)
