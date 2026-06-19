import pandas as pd
import uuid
import random
import os

# Input file paths: can be .csv or .xlsx
input_files = {
    "chat_interface": "chat_interface.xlsx",         # or .csv
    "gpto1_api": "gpto1_api.xlsx",                   # or .csv
    "gpto1_api_lightrag": "gpto1_api_lightrag.xlsx"  # or .csv
}

# Output file paths
output_public_csv = "public_scrambled_responses.csv"
output_key_csv = "key_mapping.csv"

# Helper function to read CSV or Excel
def read_file(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".csv":
        return pd.read_csv(filepath)
    elif ext in [".xls", ".xlsx"]:
        return pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

# Store all rows
all_entries = []

# Process all input files
for method, filepath in input_files.items():
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing file: {filepath}")
    
    df = read_file(filepath)

    for idx, row in df.iterrows():
        question_text = row["Question"]
        question_id = f"{method}_Q{idx+1}"

        for i in range(1, 4):  # Loop over Answer 1–3
            answer_col = f"Answer {i}"
            if answer_col not in row or pd.isna(row[answer_col]):
                continue

            response_text = row[answer_col]
            response_id = str(uuid.uuid4())

            all_entries.append({
                "response_id": response_id,
                "question_id": question_id,
                "question": question_text,
                "response": response_text,
                "method": method
            })

# Shuffle to reduce ordering bias
random.shuffle(all_entries)

# Output public CSV for MTurk
public_df = pd.DataFrame(all_entries)[["question", "response", "response_id"]]
public_df.to_csv(output_public_csv, index=False)

# Output key CSV for internal mapping
key_df = pd.DataFrame(all_entries)[["response_id", "question_id", "method"]]
key_df.to_csv(output_key_csv, index=False)

print(f"✅ Done. Created:\n- {output_public_csv}\n- {output_key_csv}")
