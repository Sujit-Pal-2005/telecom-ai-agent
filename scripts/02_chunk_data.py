import os
import json
import pandas as pd


# CONFIGURATION

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_logs.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "chunks", "text_chunks.json")

CHUNK_SIZE = 20  


# CONVERT ROW TO TEXT

def row_to_text(row):
    return f"""
Region: {row['Region']}
Tower ID: {row['Tower_ID']}
Date: {row['Date']}
Call Drops: {row['Call_Drops']}
Signal Strength: {row['Signal_Strength']}
Congestion Level: {row['Congestion_Level']}
Handoff Failure: {row['Handoff_Failure']}
Notes: {row['Notes']}
""".strip()


# CHUNKING FUNCTION

def create_chunks(text_list, chunk_size):
    chunks = []
    for i in range(0, len(text_list), chunk_size):
        chunk = "\n\n".join(text_list[i:i+chunk_size])
        chunks.append(chunk)
    return chunks


# MAIN FUNCTION

def chunk_telecom_logs(input_path, output_path):
    print("Loading cleaned dataset...")

    df = pd.read_csv(input_path)

    print(f"Total rows: {len(df)}")

    # Convert each row into structured text
    text_documents = df.apply(row_to_text, axis=1).tolist()

    print(f"Converted {len(text_documents)} rows into text format.")

    # Create chunks
    chunks = create_chunks(text_documents, CHUNK_SIZE)

    print(f"Total chunks created: {len(chunks)}")

    # Create directory if not exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save as JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=4)

    print("Chunked data saved successfully.")
    print(f"Saved at: {output_path}")



if __name__ == "__main__":
    chunk_telecom_logs(INPUT_PATH, OUTPUT_PATH)