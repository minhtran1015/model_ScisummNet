import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# File paths
DOCS_FILE = "data/raw/train_docs_v2.json"
CITATIONS_FILE = "data/raw/train_citations.json"
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
        sentence_map = {}  # Maps index -> sentence text
        index = 0
        for sec_content in sections.values():
            for sent in sec_content.get("sentences", []):
                all_sentences.append(sent["text"])
                sentence_map[index] = sent["text"]
                index += 1

        # Extract Cited Text Spans
        cited_texts = []
        if paper_id in citations_data and all_sentences:
            vectorizer = TfidfVectorizer(stop_words="english")
            tfidf_matrix = vectorizer.fit_transform(all_sentences)

            for citation in citations_data[paper_id]:
                cite_text = citation.get("cite_text", "").strip()
                if not cite_text:
                    continue

                # Compute similarity
                cite_vector = vectorizer.transform([cite_text])
                similarities = cosine_similarity(cite_vector, tfidf_matrix)[0]
                top_indices = np.argsort(similarities)[-2:][::-1]  # Get top 2 most similar sentences

                for idx in top_indices:
                    sent_text = sentence_map[idx]
                    if sent_text not in cited_texts:
                        cited_texts.append(sent_text)

        # Combine abstract and cited text spans
        combined_sentences = list(set(abstract_sentences + cited_texts))

        # Construct adjacency matrix
        adjacency_matrix = construct_sentence_graph(combined_sentences)

        # 🚨 Ensure adjacency matrix matches sentence count
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
