'''A5. Expand the above functions to
 built your own Decision Tree module.'''
# A5 - Custom Decision Tree Implementation
import pandas as pd
import numpy as np
from collections import Counter

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("top_2_hdd_models.csv")

# -----------------------------
# Utility Functions
# -----------------------------
def entropy(y):
    """Calculate entropy of a target column"""
    from math import log2
    counts = Counter(y)
    total = len(y)
    return -sum((c / total) * log2(c / total) for c in counts.values() if c > 0)


def information_gain(y, x):
    """Compute information gain for a feature x"""
    base_entropy = entropy(y)
    values = set(x)
    weighted_entropy = 0
    for v in values:
        subset_y = y[x == v]
        weighted_entropy += (len(subset_y) / len(y)) * entropy(subset_y)
    return base_entropy - weighted_entropy


def bin_numeric_feature(series, bins=3, method="frequency"):
    """Discretize numeric features into categorical bins"""
    if method == "equal_width":
        return pd.cut(series, bins=bins, labels=False, duplicates="drop")
    elif method == "frequency":
        return pd.qcut(series, q=bins, labels=False, duplicates="drop")
    else:
        raise ValueError("Invalid binning method")


# -----------------------------
# Decision Tree Node
# -----------------------------
class DecisionTreeNode:
    def __init__(self, feature=None, children=None, label=None):
        self.feature = feature
        self.children = children if children else {}
        self.label = label

    def is_leaf(self):
        return self.label is not None


# -----------------------------
# Build Decision Tree
# -----------------------------
def build_tree(X, y, depth=0, max_depth=3, bins=3):
    indent = "  " * depth  # pretty printing

    # Case 1: pure class → leaf node
    if len(set(y)) == 1:
        print(f"{indent}Leaf → Class = {y.iloc[0]}")
        return DecisionTreeNode(label=y.iloc[0])

    # Case 2: no features left or depth limit
    if X.empty or depth >= max_depth:
        majority_class = y.mode()[0]
        print(f"{indent}Leaf (majority) → Class = {majority_class}")
        return DecisionTreeNode(label=majority_class)

    # Discretize numeric columns before computing IG
    X_binned = X.copy()
    for col in X_binned.columns:
        if pd.api.types.is_numeric_dtype(X_binned[col]):
            X_binned[col] = bin_numeric_feature(X_binned[col], bins=bins, method="frequency")

    # Compute IG for all features
    gains = {col: information_gain(y, X_binned[col]) for col in X_binned.columns}
    best_feature = max(gains, key=gains.get)

    print(f"{indent}Split on: {best_feature} (Info Gain = {gains[best_feature]:.4f})")

    # Create root node
    root = DecisionTreeNode(feature=best_feature)

    # Split into branches
    for val in X_binned[best_feature].dropna().unique():
        print(f"{indent}  Branch: {best_feature} = {val}")
        mask = X_binned[best_feature] == val
        if mask.sum() == 0:
            # Empty branch → fallback to majority class
            majority_class = y.mode()[0]
            print(f"{indent}    Leaf (empty branch) → Class = {majority_class}")
            root.children[val] = DecisionTreeNode(label=majority_class)
        else:
            child = build_tree(X.drop(columns=[best_feature])[mask], y[mask], depth+1, max_depth, bins)
            root.children[val] = child

    return root


# -----------------------------
# Run Example
# -----------------------------
if __name__ == "__main__":
    # Pick a few SMART features for demo
    features = ["smart_5_raw", "smart_187_normalized", "smart_1_normalized"]
    X = df[features]
    y = df["failure"]

    print("\nBuilding custom decision tree...\n")
    tree = build_tree(X, y, max_depth=3, bins=3)
