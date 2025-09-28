import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
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

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Numeric Preprocessing
numeric_features = X.columns.tolist()
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
preprocessor = ColumnTransformer(
    transformers=[("num", numeric_transformer, numeric_features)]
)

# Pipeline with simple Logistic Regression
clf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=500))
])

# Train & Evaluate
clf_pipeline.fit(X_train, y_train)
y_pred = clf_pipeline.predict(X_test)

print("Pipeline Accuracy:", accuracy_score(y_test, y_pred))
print("\nPipeline Report:\n", classification_report(y_test, y_pred))
