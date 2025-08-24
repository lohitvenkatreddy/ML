import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
    has_xgb = True
except ImportError:
    has_xgb = False

try:
    from catboost import CatBoostClassifier
    has_cat = True
except ImportError:
    has_cat = False

# Load data
df = pd.read_csv("final_dataset.csv")

# Drop ID/leakage columns
drop_cols = [c for c in ["date","serial_number","model","datacenter","cluster_id",
                         "vault_id","pod_id","pod_slot_num","failure"] if c in df.columns]
df = df.drop(columns=drop_cols)

# Drop all-NaN columns
df = df.dropna(axis=1, how="all")

target_col = "failure_soon"
num_cols = [c for c in df.columns if c != target_col and np.issubdtype(df[c].dtype, np.number)]
X, y = df[num_cols], df[target_col].astype(int)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# Preprocess
preprocess = ColumnTransformer([("num", SimpleImputer(strategy="median"), num_cols)], remainder="drop")

# Classifiers
models = {
    "DecisionTree": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced_subsample"),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "GaussianNB": GaussianNB(),
    "MLP": MLPClassifier(max_iter=300, random_state=42),
    "SVM": SVC(probability=True, random_state=42)
}
if has_xgb: models["XGBoost"] = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
if has_cat: models["CatBoost"] = CatBoostClassifier(verbose=0, random_state=42)

# Evaluate models
rows = []
for name, clf in models.items():
    pipe = Pipeline([("prep", preprocess), ("clf", clf)])
    pipe.fit(X_train, y_train)
    y_tr, y_te = pipe.predict(X_train), pipe.predict(X_test)

    def auc_safe():
        try: return roc_auc_score(y_test, pipe.predict_proba(X_test)[:,1])
        except: return np.nan

    rows.append({
        "Model": name,
        "Train_Accuracy": accuracy_score(y_train, y_tr),
        "Train_Precision": precision_score(y_train, y_tr, zero_division=0),
        "Train_Recall": recall_score(y_train, y_tr, zero_division=0),
        "Train_F1": f1_score(y_train, y_tr, zero_division=0),
        "Test_Accuracy": accuracy_score(y_test, y_te),
        "Test_Precision": precision_score(y_test, y_te, zero_division=0),
        "Test_Recall": recall_score(y_test, y_te, zero_division=0),
        "Test_F1": f1_score(y_test, y_te, zero_division=0),
        "Test_ROC_AUC": auc_safe()
    })

results = pd.DataFrame(rows)
print(results.round(4))
results.to_csv("classifier_comparison_results.csv", index=False)
print("\n Results saved to classifier_comparison_results.csv")
