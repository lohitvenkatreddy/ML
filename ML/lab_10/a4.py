# A4: Sequential Feature Selection (Optimized)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import accuracy_score

def sequential_selection(csv_path: str, target_col: str = None):
    df = pd.read_csv(csv_path)
    if target_col is None:
        target_col = df.columns[-1]

    X = df.drop(columns=[target_col])
    y = df[target_col]
    if y.dtype == 'object':
        y = pd.factorize(y)[0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ✅ Use lightweight model instead of RandomForest
    model = LogisticRegression(max_iter=500)

    # ✅ Reduce number of features to select (optional)
    n_select = min(10, X.shape[1])  # cap at 10 for speed

    # ✅ Reduce CV folds to 2 for faster run
    sfs = SequentialFeatureSelector(
        model,
        n_features_to_select=n_select,
        direction='forward',
        cv=2,
        n_jobs=-1
    )

    print("⏳ Running Sequential Feature Selection... This may take a minute.")
    sfs.fit(X_train, y_train)
    print("✅ Feature Selection Done!")

    selected_features = X.columns[sfs.get_support()]
    model.fit(X_train[selected_features], y_train)
    y_pred = model.predict(X_test[selected_features])
    acc = accuracy_score(y_test, y_pred)

    print(f"A4 - Sequential Selection Accuracy: {acc:.4f}")
    print(f"Selected Features ({len(selected_features)}): {list(selected_features)}")

if __name__ == '__main__':
    sequential_selection('train.csv')
