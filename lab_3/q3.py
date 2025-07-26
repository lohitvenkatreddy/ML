import pandas as pd
import numpy as np
from scipy.spatial.distance import minkowski
import matplotlib.pyplot as plt

def compute_minkowski_distances(vec1, vec2, r_max=10):
    r_vals = list(range(1, r_max + 1))
    distances = [minkowski(vec1, vec2, p=r) for r in r_vals]
    return r_vals, distances

# Load data
df = pd.read_csv("Hyderbad_House_price.csv")

# Extract BHK from title and keep only 2BHK and 3BHK rows
df['bhk'] = df['title'].str.extract(r'(\d) BHK').astype(float)
df = df[df['bhk'].isin([2, 3])]

# ✅ Define features
features = ['price(L)', 'rate_persqft', 'area_insqft']

# Select features
X = df[features].values

# Pick two sample vectors
vec1, vec2 = X[0], X[1]

# Compute Minkowski distances
r_values, dist_values = compute_minkowski_distances(vec1, vec2)

# Plot
plt.plot(r_values, dist_values, marker='o')
plt.title("Minkowski Distance for r = 1 to 10")
plt.xlabel("r")
plt.ylabel("Distance")
plt.grid(True)
plt.show()

# Print results
print("=== A3: Minkowski Distance ===")
for r, dist in zip(r_values, dist_values):
    print(f"r = {r}: Distance = {dist:.2f}")
