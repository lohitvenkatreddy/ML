import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
import lime
import lime.lime_tabular

# -------------------------
# Load dataset
# -------------------------
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

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# -------------------------
# Preprocessing
# -------------------------
numeric_features = X.columns.tolist()
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[("num", numeric_transformer, numeric_features)]
)

# -------------------------
# Model (Logistic Regression for simplicity)
# -------------------------
clf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=500))
])

clf_pipeline.fit(X_train, y_train)

# -------------------------
# LIME Explainer
# -------------------------
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=numeric_features,
    class_names=["No Failure", "Failure"],
    mode="classification"
)

# Select one instance to explain
i = 5
sample = pd.DataFrame([X_test.iloc[i]], columns=numeric_features)

exp = explainer.explain_instance(
    data_row=sample.values[0],
    predict_fn=lambda x: clf_pipeline.predict_proba(pd.DataFrame(x, columns=numeric_features)),
    num_features=10
)

# -------------------------
# Save explanation to HTML (works outside Jupyter)
# -------------------------
html_file = "lime_explanation.html"
exp.save_to_file(html_file)
print(f"LIME explanation saved to {html_file}")
