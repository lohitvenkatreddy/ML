import pandas as pd
from collections import Counter

df = pd.read_csv("top_2_hdd_models.csv")

def entropy(y):
    from math import log2
    counts = Counter(y)
    total = len(y)
    return -sum((c/total) * log2(c/total) for c in counts.values() if c > 0)

def information_gain(y, x):
    base = entropy(y)
    vals = set(x)
    weighted = 0
    for v in vals:
        subset_y = y[x == v]
        weighted += (len(subset_y)/len(y)) * entropy(subset_y)
    return base - weighted

class DecisionTreeNode:
    def __init__(self, feature=None, children=None, label=None):
        self.feature = feature
        self.children = children if children else {}
        self.label = label

def build_tree(X, y, depth=0):
    indent = "  " * depth  # for pretty printing
    
    # Case 1: pure class (leaf node)
    if len(set(y)) == 1:
        print(f"{indent}Leaf node → Class = {y.iloc[0]}")
        return DecisionTreeNode(label=y.iloc[0])
    
    # Case 2: no features left
    if X.empty:
        majority_class = y.mode()[0]
        print(f"{indent}Leaf node (no features) → Class = {majority_class}")
        return DecisionTreeNode(label=majority_class)
    
    # Compute information gain for all features
    gains = {col: information_gain(y, X[col]) for col in X.columns}
    best_feature = max(gains, key=gains.get)
    
    print(f"{indent}Splitting on feature: {best_feature} (Info Gain = {gains[best_feature]:.4f})")
    
    root = DecisionTreeNode(feature=best_feature)
    
    # Create children for each value of the best feature
    for val in X[best_feature].unique():
        print(f"{indent}  Branch: {best_feature} = {val}")
        mask = X[best_feature] == val
        child = build_tree(X[mask].drop(columns=[best_feature]), y[mask], depth+1)
        root.children[val] = child
    
    return root

if __name__ == "__main__":
    # Example: take a smaller subset to test
    X = df.drop(columns=['failure']).iloc[:50]  # reduce for debug readability
    y = df['failure'].iloc[:50]
    
    print("Building decision tree...\n")
    tree = build_tree(X, y)
