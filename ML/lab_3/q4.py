import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("Hyderbad_House_price.csv")

# Extract BHK from title and filter to 2BHK and 3BHK only
df['bhk'] = df['title'].str.extract(r'(\d) BHK').astype(float)
df = df[df['bhk'].isin([2, 3])]

# ✅ Define features to be used
features = ['price(L)', 'rate_persqft', 'area_insqft']
X = df[features].values
y = df['bhk'].values  # class labels

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Print results
print("=== A4: Train-Test Split ===")
print("Training set size:", X_train.shape[0])
print("Testing set size:", X_test.shape[0])
