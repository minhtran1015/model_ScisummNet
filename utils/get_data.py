import json

DOCS_FILE = "data/raw/train_docs_v2.json"
CITATIONS_FILE = "data/raw/train_citations.json"
LABEL_FILE = "data/raw/train_label.json"
EXTRACTED_FILE = "data/processed/extracted_sentences.json"

def get_docs_data():
    with open(DOCS_FILE, "r") as f:
        docs_data = json.load(f)
    return docs_data

def get_citations_data():
    with open(CITATIONS_FILE, "r") as f:
        citations_data = json.load(f)
    return citations_data

def get_label_data():
    with open(DOCS_FILE, "r") as f:
        label_data = json.load(f)
    return label_data

def get_extracted_data():
    with open(EXTRACTED_FILE, "r") as f:
        extracted_data = json.load(f)
    return extracted_data