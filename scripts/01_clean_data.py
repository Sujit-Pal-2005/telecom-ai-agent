import os
import pandas as pd


# CONFIGURATION

RAW_DATA_PATH = "data/raw/telecom_logs.csv"
PROCESSED_DATA_PATH = "data/processed/cleaned_logs.csv"


# CLEANING FUNCTION
def clean_telecom_data(input_path: str, output_path: str):
    print("Loading raw dataset...")

    df = pd.read_csv(input_path)

    print(f"Initial rows: {len(df)}")

    #Remove completely empty rows
    df.dropna(how="all", inplace=True)

    # Remove duplicate rows
    df.drop_duplicates(inplace=True)

    # Handle missing values
    df.fillna({
        "Call_Drops": 0,
        "Signal_Strength": "-100 dBm",
        "Congestion_Level": "Unknown",
        "Handoff_Failure": "0%",
        "Notes": "No remarks"
    }, inplace=True)

    # Standardize text columns
    text_columns = [
        "Region",
        "Tower_ID",
        "Congestion_Level",
        "Notes"
    ]

    for col in text_columns:
        df[col] = df[col].astype(str).str.strip()

    #Convert Date column to proper format
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Remove invalid dates
    df = df[df["Date"].notna()]

    #  Convert numeric fields properly
    df["Call_Drops"] = pd.to_numeric(df["Call_Drops"], errors="coerce").fillna(0)

    # Final shape
    print(f"Rows after cleaning: {len(df)}")

    # Create output directory if not exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save cleaned file
    df.to_csv(output_path, index=False)

    print("Cleaned dataset saved successfully.")
    print(f"Saved at: {output_path}")


# MAIN EXECUTION
if __name__ == "__main__":
    clean_telecom_data(RAW_DATA_PATH, PROCESSED_DATA_PATH)