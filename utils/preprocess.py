import json
import os
import re
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt")

# Paths to files
DOCS_FILE = "data/raw/test_docs_v5.json"
CITATIONS_FILE = "data/raw/test_citations.json"
LABELS_FILE = "data/raw/test_label.json"
OUTPUT_FILE = "data/processed/preprocessed_data.json"

def clean_text(text):
    """Remove special characters and extra spaces."""
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

def extract_paper_data(docs_data):
    """Extract abstract + conclusion from each paper."""
    extracted_data = {}

    for paper_id, content in docs_data.items():
        sections = content.get("sections", {})

        # Extract Abstract and Conclusion if available
        abstract = " ".join([s["text"] for s in sections.get("Abstract", {}).get("sentences", [])])
        conclusion = " ".join([s["text"] for s in sections.get("Conclusion", {}).get("sentences", [])])

        # Store extracted text
        extracted_data[paper_id] = {
            "abstract": clean_text(abstract),
            "conclusion": clean_text(conclusion),
        }

    return extracted_data

def extract_citation_data(citations_data):
    """Extract citation-reference pairs."""
    citation_pairs = {}

    for paper_id, citations in citations_data.items():
        citation_pairs[paper_id] = []
        for citation in citations:
            cite_text = citation.get("cite_text", "")
            refer_text = citation.get("refer_text", "")
            citation_pairs[paper_id].append({
                "cite_text": clean_text(cite_text),
                "refer_text": clean_text(refer_text),
            })

    return citation_pairs

def extract_labels(labels_data):
    """Extract gold summaries."""
    return {paper_id: [" ".join(label)] for paper_id, label in labels_data.items()}

def preprocess_data():
    """Combined function to process all datasets."""
    with open(DOCS_FILE, "r", encoding="utf-8") as f:
        docs_data = json.load(f)

    with open(CITATIONS_FILE, "r", encoding="utf-8") as f:
        citations_data = json.load(f)

    with open(LABELS_FILE, "r", encoding="utf-8") as f:
        labels_data = json.load(f)

    # Process data
    papers = extract_paper_data(docs_data)
    citations = extract_citation_data(citations_data)
    summaries = extract_labels(labels_data)

    # Combine into final dataset
    final_data = {}
    for paper_id in papers.keys():
        final_data[paper_id] = {
            "abstract": papers[paper_id]["abstract"],
            "conclusion": papers[paper_id]["conclusion"],
            "citations": citations.get(paper_id, []),
            "summary": summaries.get(paper_id, [""]),
        }

    # Save processed data
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=4)

    print(f"✅ Processed data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    preprocess_data()
