import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Load data
df = pd.read_csv("final_dataset.csv")

# Drop ID/leakage columns
drop_cols = [c for c in ["date","serial_number","model","datacenter","cluster_id",
                         "vault_id","pod_id","pod_slot_num","failure"] if c in df.columns]
df = df.drop(columns=drop_cols)

# Drop all-NaN columns (important!)
df = df.dropna(axis=1, how="all")

target_col = "failure_soon"
num_cols = [c for c in df.columns if c != target_col and np.issubdtype(df[c].dtype, np.number)]
X, y = df[num_cols], df[target_col].astype(int)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# Preprocess: impute remaining missing values
preprocess = ColumnTransformer([("num", SimpleImputer(strategy="median"), num_cols)], remainder="drop")

# Model
rf = RandomForestClassifier(class_weight="balanced_subsample", random_state=42)

param_dist = {
    "clf__n_estimators": [100, 200, 300],
    "clf__max_depth": [None, 10, 20, 30],
    "clf__min_samples_split": [2, 5, 10],
    "clf__min_samples_leaf": [1, 2, 4]
}

pipe = Pipeline([("prep", preprocess), ("clf", rf)])

# RandomizedSearchCV
search = RandomizedSearchCV(
    pipe, param_distributions=param_dist, n_iter=10,
    scoring="roc_auc", cv=3, n_jobs=-1, random_state=42, verbose=2
)
search.fit(X_train, y_train)

print(" Best params:", search.best_params_)
print(" Test ROC AUC:", roc_auc_score(y_test, search.predict_proba(X_test)[:,1]))
print(" Classification report:\n", classification_report(y_test, search.predict(X_test)))
