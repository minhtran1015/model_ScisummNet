import json
from datasets import Dataset

DATA_FILE = "data/processed/preprocessed_data.json"

# Load processed JSON
with open(DATA_FILE, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# Convert to Hugging Face Dataset format
formatted_data = []
for paper_id, content in raw_data.items():
    formatted_data.append({
        "input_text": f"Summarize: {content['abstract']} {content['conclusion']}",
        "target_text": content["summary"][0]
    })

dataset = Dataset.from_list(formatted_data)
dataset = dataset.train_test_split(test_size=0.1)

# Save dataset for training
dataset.save_to_disk("data/processed/hf_dataset")
print("✅ Dataset saved to data/processed/hf_dataset")
