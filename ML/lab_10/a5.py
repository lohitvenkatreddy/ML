# A5: Explainable AI (LIME and SHAP)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import shap
from lime.lime_tabular import LimeTabularExplainer
import numpy as np

def explain_with_lime_and_shap(csv_path: str, target_col: str = None):
    df = pd.read_csv(csv_path)
    if target_col is None:
        target_col = df.columns[-1]

    X = df.drop(columns=[target_col])
    y = pd.factorize(df[target_col])[0]

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # --- LIME ---
    lime_exp = LimeTabularExplainer(
        X_train,
        feature_names=X.columns,
        class_names=[f'class_{i}' for i in np.unique(y)],
        mode='classification'
    )
    exp = lime_exp.explain_instance(X_test[0], model.predict_proba, num_features=10)
    exp.save_to_file('A5_LIME_Explanation.html')

    # --- SHAP ---
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # summary plot
    shap.summary_plot(shap_values, X_test, feature_names=X.columns, show=False)
    plt.title("SHAP Summary Plot (X_test)")
    plt.tight_layout()
    plt.savefig('A5_SHAP_Summary.png', dpi=200)
    plt.close()

    # --- Robust Force Plot Handling ---
    shap.initjs()
    try:
        # Case 1: Binary classification → list of arrays
        if isinstance(shap_values, list):
            idx = 1 if len(shap_values) > 1 else 0
            shap.force_plot(
                explainer.expected_value[idx],
                shap_values[idx][0, :],
                X_test[0, :],
                matplotlib=True
            )
        else:
            # Case 2: Single array (regression or multiclass)
            shap.force_plot(
                explainer.expected_value,
                shap_values[0, :],
                X_test[0, :],
                matplotlib=True
            )

        plt.tight_layout()
        plt.savefig('A5_SHAP_ForcePlot.png', dpi=200)
        plt.close()
    except Exception as e:
        print(f"⚠️ Could not generate SHAP force plot: {e}")
    
    print("✅ LIME and SHAP explanations generated successfully!\nFiles saved: \n- A5_LIME_Explanation.html\n- A5_SHAP_Summary.png\n- A5_SHAP_ForcePlot.png (if supported)")

if __name__ == '__main__':
    explain_with_lime_and_shap('train.csv')