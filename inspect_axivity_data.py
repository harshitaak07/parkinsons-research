import pandas as pd
from src.preprocessing.axivity_gait import preprocess_axivity

def main():
    df = preprocess_axivity('data/raw/Gait_Data___Arm_swing__Axivity__06Sep2025.csv', save=False)
    print(f"Gait data shape: {df.shape}")
    numeric_df = df.select_dtypes(include='number')
    print(f"Numeric columns count: {len(numeric_df.columns)}")
    print("Sample numeric data:")
    print(numeric_df.head())

if __name__ == "__main__":
    main()
