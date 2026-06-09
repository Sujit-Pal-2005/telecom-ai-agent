import os
import re
import pandas as pd
import mysql.connector
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "telecom_db")

RAW_DATASET = "data/log/dataset.csv"


def clean_signal_strength(val):
    if pd.isna(val):
        return -100

    match = re.search(r"(-?\d+)", str(val))
    if match:
        return int(match.group(1))

    return -100


def clean_handoff_failure(val):
    if pd.isna(val):
        return 0

    match = re.search(r"(\d+)", str(val))
    if match:
        return int(match.group(1))

    return 0


def clean_data():
    print("Reading dataset...")

    df = pd.read_csv(RAW_DATASET)

    print(f"Original rows: {len(df)}")

    # Remove duplicates and empty rows
    df.dropna(how="all", inplace=True)
    df.drop_duplicates(inplace=True)

    print(f"Rows after cleaning: {len(df)}")

    # Standardize text fields
    df["Region"] = (
        df["Region"]
        .astype(str)
        .str.strip()
        .str.title()
    )

    df["Region"] = df["Region"].replace({
        "Bangalore": "Bengaluru"
    })

    df["Tower_ID"] = (
        df["Tower_ID"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["Congestion_Level"] = (
        df["Congestion_Level"]
        .astype(str)
        .str.strip()
        .str.capitalize()
    )

    df["Notes"] = (
        df["Notes"]
        .fillna("No remarks")
        .astype(str)
        .str.strip()
    )

    # Date cleanup
    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df = df[df["Date"].notna()]

    # Numeric cleanup
    df["Call_Drops"] = (
        pd.to_numeric(
            df["Call_Drops"],
            errors="coerce"
        )
        .fillna(0)
        .astype(int)
    )

    df["Signal_Strength"] = (
        df["Signal_Strength"]
        .apply(clean_signal_strength)
        .astype(int)
    )

    df["Handoff_Failure"] = (
        df["Handoff_Failure"]
        .apply(clean_handoff_failure)
        .astype(int)
    )

    print("Data cleaning completed.")

    return df


def create_database():
    print(
        f"Connecting to MySQL server at "
        f"{DB_HOST}:{DB_PORT}..."
    )

    conn = mysql.connector.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD
    )

    cursor = conn.cursor()

    cursor.execute(
        f"CREATE DATABASE IF NOT EXISTS {DB_NAME}"
    )

    conn.commit()

    cursor.close()
    conn.close()

    print(
        f"Database '{DB_NAME}' created or verified."
    )


def ingest_to_mysql():
    df = clean_data()

    create_database()

    db_uri = (
        f"mysql+mysqlconnector://"
        f"{DB_USER}:{DB_PASSWORD}"
        f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

    engine = create_engine(db_uri)

    print(
        "Writing records to "
        "'network_metrics' table..."
    )

    df.to_sql(
        name="network_metrics",
        con=engine,
        if_exists="replace",
        index=False,
        chunksize=1000
    )

    print("Creating indexes...")

    with engine.connect() as con:

        con.execute(text("""
            CREATE INDEX idx_region
            ON network_metrics (Region(50))
        """))

        con.execute(text("""
            CREATE INDEX idx_tower
            ON network_metrics (Tower_ID(20))
        """))

        con.execute(text("""
            CREATE INDEX idx_date
            ON network_metrics (Date)
        """))

        con.commit()

    print("\nDatabase ingestion completed successfully!")

    print("\nTable: network_metrics")

    print("Columns:")
    print("- Region")
    print("- Tower_ID")
    print("- Date")
    print("- Call_Drops")
    print("- Signal_Strength")
    print("- Congestion_Level")
    print("- Handoff_Failure")
    print("- Notes")


if __name__ == "__main__":
    try:
        ingest_to_mysql()

    except Exception as e:
        print(
            f"Error during database ingestion: {e}"
        )