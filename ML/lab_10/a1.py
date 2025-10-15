# A1: Feature Correlation Analysis
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def feature_correlation_heatmap(csv_path: str, target_col: str = None):
    df = pd.read_csv(csv_path)
    if target_col is None:
        target_col = df.columns[-1]

    corr = df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=False, cmap='coolwarm')
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('A1_correlation_heatmap.png', dpi=200)
    plt.close()

if __name__ == '__main__':
    feature_correlation_heatmap('train.csv')
