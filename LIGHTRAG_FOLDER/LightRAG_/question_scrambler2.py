# FORMAT FOR GENERATING RESPONSE DATA EXCEL OR CSV:
# QUESTION  | ANSWER 1 | ANSWER 2  | ANSWER 3



import pandas as pd
import uuid
import random
import os

# Configuration
input_files = {
    "chat_interface": "chat_interface.xlsx",
    "gpto1_api": "gpto1_api.xlsx",
    "gpto1_api_lightrag": "gpto1_api_lightrag.xlsx"
}
output_public_csv = "public_scrambled_responses.csv"
output_key_csv = "key_mapping.csv"
max_consecutive = 3  # No more than 3 responses in a row from the same method
max_attempts = 1000  # To avoid infinite loops

# Helper to read csv or xlsx
def read_file(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".csv":
        return pd.read_csv(filepath)
    elif ext in [".xls", ".xlsx"]:
        return pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

# Gather all entries
all_entries = []

for method, filepath in input_files.items():
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing file: {filepath}")
    
    df = read_file(filepath)
    for idx, row in df.iterrows():
        question_text = row["Question"]
        question_id = f"{method}_Q{idx+1}"
        for i in range(1, 4):
            answer_col = f"Answer {i}"
            if answer_col not in df.columns or pd.isna(row[answer_col]):
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

# Helper to check max-consecutive constraint
def is_well_mixed(seq, key, max_consecutive):
    count = 1
    for i in range(1, len(seq)):
        if seq[i][key] == seq[i - 1][key]:
            count += 1
            if count > max_consecutive:
                return False
        else:
            count = 1
    return True

# Shuffle with constraint
for attempt in range(max_attempts):
    random.shuffle(all_entries)
    if is_well_mixed(all_entries, "method", max_consecutive):
        break
else:
    raise RuntimeError("Failed to create well-mixed sequence after many attempts.")

# Save public CSV
public_df = pd.DataFrame(all_entries)[["question", "response", "response_id"]]
public_df.to_csv(output_public_csv, index=False)

# Save key CSV
key_df = pd.DataFrame(all_entries)[["response_id", "question_id", "method"]]
key_df.to_csv(output_key_csv, index=False)

print("✅ Done with well-mixed sequence.")
