import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def classify_customers(df):
    df['Class'] = df['Payment (Rs)'].apply(lambda x: 'RICH' if x > 200 else 'POOR')
    X = df[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']]
    y = df['Class'].map({'POOR': 0, 'RICH': 1})
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    return report, df

def main():
    input_file = "LabData.xlsx"
    sheet_name = "Purchase data"
    df = pd.read_excel(input_file, sheet_name=sheet_name)
    report, updated_df = classify_customers(df)
    print("A2 Results: Classification Report")
    print(report)
    with pd.ExcelWriter(input_file, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        updated_df.to_excel(writer, sheet_name=sheet_name, index=False)

main()