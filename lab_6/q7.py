import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("top_2_hdd_models.csv")

# Target variable
y = df["failure"]

# Pick two features for visualization
features = ["smart_5_raw", "smart_187_raw"]  # change if needed
X = df[features].copy()

# Encode if non-numeric
for col in X.columns:
    if X[col].dtype == "object":
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

# Train decision tree classifier
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=42)
clf.fit(X, y)

print("Training accuracy:", clf.score(X, y))

# -------- Plot Decision Boundaries -------- #
x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1

# Reduce step size to avoid huge memory usage
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(12, 6))

# Decision boundary plot
plt.subplot(1, 2, 1)
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, edgecolor="k", s=40, cmap=plt.cm.coolwarm)
plt.xlabel(features[0])
plt.ylabel(features[1])
plt.title("Decision Tree Decision Boundaries")

# Tree plot
plt.subplot(1, 2, 2)
plot_tree(clf, feature_names=features, class_names=[str(c) for c in clf.classes_],
          filled=True, rounded=True, fontsize=8)
plt.title("Decision Tree Structure")

plt.tight_layout()
plt.show()
