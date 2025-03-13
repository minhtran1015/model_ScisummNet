import json

DOCS_FILE = "data/raw/train_docs_v2.json"
CITATIONS_FILE = "data/raw/train_citations.json"
LABEL_FILE = "data/raw/train_label.json"

TEST_DOCS_FILE = "data/raw/test_docs_v5.json"
TEST_CITATIONS_FILE = "data/raw/test_citations.json"
TEST_LABEL_FILE = "data/raw/test_label.json"

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

def get_test_docs_data():
    with open(TEST_DOCS_FILE, "r") as f:
        test_docs_data = json.load(f)
    return test_docs_data

def get_test_citations_data():
    with open(TEST_CITATIONS_FILE, "r") as f:
        test_citations_data = json.load(f)
    return test_citations_data

def get_test_label_data():
    with open(TEST_LABEL_FILE, "r") as f:
        test_label_data = json.load(f)
    return test_label_data

def get_extracted_data():
    with open(EXTRACTED_FILE, "r") as f:
        extracted_data = json.load(f)
    return extracted_data