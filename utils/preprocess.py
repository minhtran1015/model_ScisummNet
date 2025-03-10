import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# File paths
DOCS_FILE = "data/raw/test_docs_v5.json"
CITATIONS_FILE = "data/raw/test_citations.json"
OUTPUT_FILE = "data/processed/extracted_sentences.json"

def construct_sentence_graph(sentences):
    """Construct sentence relation graph using TF-IDF cosine similarity."""
    if not sentences or len(sentences) < 2:
        return np.eye(len(sentences)).tolist()  # Return identity matrix if <2 sentences

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(sentences)

    if tfidf_matrix.shape[0] != len(sentences):  # 🚨 Check for missing rows
        print(f"🚨 TF-IDF shape mismatch: {tfidf_matrix.shape[0]} vectors for {len(sentences)} sentences")

    adjacency_matrix = cosine_similarity(tfidf_matrix)
    np.fill_diagonal(adjacency_matrix, 1)  # Self-connections
    return adjacency_matrix.tolist()

def extract_sentences_for_summarization():
    """Extracts abstracts and cited text spans from the dataset."""
    with open(DOCS_FILE, "r", encoding="utf-8") as f:
        docs_data = json.load(f)
    with open(CITATIONS_FILE, "r", encoding="utf-8") as f:
        citations_data = json.load(f)

    extracted_data = {}

    for paper_id, content in docs_data.items():
        sections = content.get("sections", {})

        # Extract abstract sentences
        abstract_sentences = [s["text"] for s in sections.get("Abstract", {}).get("sentences", [])]

        # Collect all sentences in the paper
        all_sentences = []
        for sec_content in sections.values():
            all_sentences.extend([s["text"] for s in sec_content.get("sentences", [])])

        # Ensure unique sentences
        combined_sentences = list(set(abstract_sentences + all_sentences))

        # Construct adjacency matrix
        adjacency_matrix = construct_sentence_graph(combined_sentences)

        # 🚨 Ensure the adjacency matrix matches sentence count
        if len(combined_sentences) != len(adjacency_matrix):
            print(f"🚨 Adjacency matrix mismatch in {paper_id}: {len(combined_sentences)} sentences vs {len(adjacency_matrix)} nodes")
            continue  # Skip this entry

        extracted_data[paper_id] = {
            "input_sentences": combined_sentences,
            "adjacency_matrix": adjacency_matrix,
        }

    # Save processed sentences
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(extracted_data, f, indent=4)

    print(f"✅ Extracted sentences and adjacency matrices saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    extract_sentences_for_summarization()
