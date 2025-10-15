'''Draw and visualize the decision tree constructed based on your data. (Refer above provided web
sources [1] & [2] for understanding and learning on how to visualize a DT).'''
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("top_2_hdd_models.csv")

# Separate features and target
X = df.drop(columns=["failure"])
y = df["failure"]

# Encode non-numeric columns
for col in X.columns:
    if X[col].dtype == "object":   # Strings like dates, models
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build Decision Tree
model = DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
print("Training accuracy:", model.score(X_train, y_train))
print("Testing accuracy:", model.score(X_test, y_test))
print(y_test[24:25])
# Plot tree
plt.figure(figsize=(25, 15))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=[str(c) for c in model.classes_],
    filled=True,
    rounded=True,
    fontsize=8
)
plt.show()
