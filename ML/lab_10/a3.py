# A3: PCA (Retaining 95% variance)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def pca_95(csv_path: str, target_col: str = None):
    df = pd.read_csv(csv_path)
    if target_col is None:
        target_col = df.columns[-1]

    X = df.drop(columns=[target_col])
    y = df[target_col]
    if y.dtype == 'object':
        y = pd.factorize(y)[0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    pca = PCA(n_components=0.95, svd_solver='full')
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_pca, y_train)
    acc = accuracy_score(y_test, model.predict(X_test_pca))
    print(f"A3 - PCA(95%) Accuracy: {acc:.4f}, Components: {pca.n_components_}")

if __name__ == '__main__':
    pca_95('train.csv')
