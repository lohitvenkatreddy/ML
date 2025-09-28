import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("train.csv")

# Drop non-informative columns
drop_cols = ["serial_number", "date", "model", "datacenter"]
df_clean = df.drop(columns=[c for c in drop_cols if c in df.columns])

# Features & Target
X = df_clean.drop(columns=["failure"])
y = df_clean["failure"]

# Drop features with too many NaNs (>70%)
valid_features = X.columns[X.isnull().mean() < 0.7]
X = X[valid_features]

# Drop zero-variance features
selector = VarianceThreshold(threshold=0.0)
X = pd.DataFrame(selector.fit_transform(X),
                 columns=X.columns[selector.get_support()])

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Preprocessing
numeric_features = X.columns.tolist()
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
preprocessor = ColumnTransformer(
    transformers=[("num", numeric_transformer, numeric_features)]
)

# Stacking Classifier
base_estimators = [
    ("lr", LogisticRegression(max_iter=500)),
    ("rf", RandomForestClassifier(n_estimators=50, random_state=42))
]
stack_clf = StackingClassifier(
    estimators=base_estimators,
    final_estimator=LogisticRegression(max_iter=500)
)

# Pipeline for training
clf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", stack_clf)
])

# Train and Evaluate
clf_pipeline.fit(X_train, y_train)
y_pred = clf_pipeline.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
