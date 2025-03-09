import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# File paths
DOCS_FILE = "data/raw/test_docs_v5.json"
CITATIONS_FILE = "data/raw/test_citations.json"
OUTPUT_FILE = "data/processed/extracted_sentences.json"


def compute_authority_scores(citations_data):
    """Compute authority scores based on citation counts."""
    authority_scores = {}
    
    # Count how many times each paper is cited
    citation_counts = {}
    for paper_id, citations in citations_data.items():
        for citation in citations:
            cite_id = citation.get("cite_ID")
            if cite_id:
                citation_counts[cite_id] = citation_counts.get(cite_id, 0) + 1
    
    # Assign authority scores
    for paper_id, citations in citations_data.items():
        authority_scores[paper_id] = sum(citation_counts.get(cite["cite_ID"], 0) for cite in citations)
    
    return authority_scores


def construct_sentence_graph(sentences):
    """Construct sentence relation graph using TF-IDF cosine similarity."""
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(sentences)
    adjacency_matrix = cosine_similarity(tfidf_matrix)
    np.fill_diagonal(adjacency_matrix, 1)  # Add self-connections
    return adjacency_matrix


def extract_sentences_for_summarization():
    """Extracts abstracts and cited text spans from the dataset with authority scores."""
    with open(DOCS_FILE, "r", encoding="utf-8") as f:
        docs_data = json.load(f)
    with open(CITATIONS_FILE, "r", encoding="utf-8") as f:
        citations_data = json.load(f)
    
    authority_scores = compute_authority_scores(citations_data)
    extracted_data = {}
    
    for paper_id, content in docs_data.items():
        sections = content.get("sections", {})
        
        # Extract abstract
        abstract_sentences = [s["text"] for s in sections.get("Abstract", {}).get("sentences", [])]
        abstract_authority = authority_scores.get(paper_id, 0)  # Use RP's citation count
        
        # Get all sentences in the reference paper
        all_sentences = []
        sentence_map = {}  # Map index to sentence text
        index = 0
        for sec_name, sec_content in sections.items():
            for sent in sec_content.get("sentences", []):
                all_sentences.append(sent["text"])
                sentence_map[index] = sent["text"]
                index += 1
        
        # Compute TF-IDF similarity for citations
        cited_texts = []
        cited_authority = {}
        if paper_id in citations_data:
            vectorizer = TfidfVectorizer(stop_words="english")
            tfidf_matrix = vectorizer.fit_transform(all_sentences)
            
            for citation in citations_data[paper_id]:
                cite_text = citation.get("cite_text", "")
                cite_id = citation.get("cite_ID")
                if not cite_text:
                    continue
                
                cite_vector = vectorizer.transform([cite_text])
                similarities = cosine_similarity(cite_vector, tfidf_matrix)[0]
                top_indices = np.argsort(similarities)[-2:][::-1]  # Get top 2 most similar sentences
                
                for idx in top_indices:
                    cited_texts.append(sentence_map[idx])
                    cited_authority[sentence_map[idx]] = authority_scores.get(cite_id, 0)
        
        # Combine abstract and cited text spans with authority scores
        combined_sentences = list(set(abstract_sentences + cited_texts))
        authority_map = {sent: cited_authority.get(sent, abstract_authority) for sent in combined_sentences}
        
        # Construct sentence relation graph
        adjacency_matrix = construct_sentence_graph(combined_sentences)
        
        extracted_data[paper_id] = {
            "input_sentences": combined_sentences,
            "authority_scores": authority_map,
            "adjacency_matrix": adjacency_matrix.tolist(),
        }
    
    # Save processed sentences
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(extracted_data, f, indent=4)
    
    print(f"✅ Extracted sentences with authority scores and sentence relation graph saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    extract_sentences_for_summarization()
