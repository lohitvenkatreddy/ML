'''For the clustering done in A4, calculate the: (i) Silhouette Score, (ii) CH Score and (iii) DB Index.
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
kmeans = KMeans(n_clusters=2, random_state=42).fit(X_train)
silhouette_score(X_train, kmeans.labels_)
calinski_harabasz_score(X_train, kmeans.labels_)
davies_bouldin_score(X_train, kmeans.labels_)'''
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def load_clustering_data(filepath, target_col, max_rows=5000, top_n_features=10):
    """Load SMART data for clustering, drop high-NaN columns and target column, reduce size."""
    df = pd.read_csv(filepath)

    # Select SMART raw attributes
    smart_cols = [col for col in df.columns if 'smart_' in col and '_raw' in col]
    df = df[smart_cols + [target_col]]

    # Drop columns with >50% missing
    df = df.loc[:, df.isnull().mean() < 0.5]

    # Drop rows with NaNs
    df = df.dropna()

    # Drop target column
    X = df.drop(columns=[target_col])

    # Keep only numeric columns
    X = X.select_dtypes(include='number')

    # Reduce to top N features by variance
    variances = X.var().sort_values(ascending=False)
    top_features = variances.head(top_n_features).index
    X = X[top_features]

    # Downsample for speed
    if len(X) > max_rows:
        X = X.sample(n=max_rows, random_state=42)

    return X

def evaluate_kmeans_clustering(X, k=2):
    """Fit KMeans and return silhouette, CH, and DB scores."""
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    labels = kmeans.labels_

    return (
        silhouette_score(X, labels),
        calinski_harabasz_score(X, labels),
        davies_bouldin_score(X, labels)
    )

if __name__ == "__main__":
    file_path = "top_2_hdd_models.csv"
    target = "failure"

    X = load_clustering_data(file_path, target, max_rows=5000, top_n_features=10)
    sil, ch, db = evaluate_kmeans_clustering(X, k=2)

    print("Silhouette Score:", sil)
    print("Calinski-Harabasz Score:", ch)
    print("Davies-Bouldin Index:", db)










# Silhouette Score: Measures how well-separated and cohesive clusters are (-1 to 1, higher is better).
# Calinski-Harabasz Score: Ratio of between-cluster dispersion to within-cluster dispersion (higher is better).
# Davies-Bouldin Index: Average similarity between clusters (lower is better).







