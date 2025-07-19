import pandas as pd

def load_data(file_path, sheet_name):
    return pd.read_excel(file_path, sheet_name=sheet_name)

def preprocess_binary_attributes(df):
    binary_columns = [col for col in df.columns if set(df[col].unique()) <= {'t', 'f', '?'}]
    df_binary = df[binary_columns].replace({'t': 1, 'f': 0, '?': 0})
    return df_binary

def calculate_similarity(vec1, vec2):
    f11 = sum((vec1 == 1) & (vec2 == 1))
    f00 = sum((vec1 == 0) & (vec2 == 0))
    f10 = sum((vec1 == 1) & (vec2 == 0))
    f01 = sum((vec1 == 0) & (vec2 == 1))
    jc = f11 / (f11 + f10 + f01) if (f11 + f10 + f01) != 0 else 0
    smc = (f11 + f00) / (f11 + f10 + f01 + f00)
    return {
        "f11": f11,
        "f00": f00,
        "f10": f10,
        "f01": f01,
        "jaccard": jc,
        "smc": smc
    }

def main():
    file_path = "LabData.xlsx"
    sheet_name = "thyroid0387_UCI"
    df = load_data(file_path, sheet_name)
    binary_df = preprocess_binary_attributes(df)
    vec1 = binary_df.iloc[0].astype(int).values
    vec2 = binary_df.iloc[1].astype(int).values
    similarity_result = calculate_similarity(vec1, vec2)
    print("---- Similarity Measures ----")
    print(f"f11: {similarity_result['f11']}, f00: {similarity_result['f00']}, "
          f"f10: {similarity_result['f10']}, f01: {similarity_result['f01']}")
    print(f"Jaccard Coefficient (JC): {similarity_result['jaccard']:.4f}")
    print(f"Simple Matching Coefficient (SMC): {similarity_result['smc']:.4f}")
    print("\nInterpretation:")
    print("- JC ignores '0-0' matches and focuses on shared '1's, making it useful for sparse binary data.")
    print("- SMC considers both '1-1' and '0-0' matches, which can be misleading for sparse binary vectors.")

if __name__ == "__main__":
    main()
