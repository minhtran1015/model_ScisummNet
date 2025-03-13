import torch
import json
import os
import sys
from rouge_score import rouge_scorer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.inference.hybrid_1 import summarize_paper

def load_test_data(docs_path, citations_path, labels_path):
    with open(docs_path, 'r', encoding='utf-8') as f:
        docs_data = json.load(f)
    with open(citations_path, 'r', encoding='utf-8') as f:
        citations_data = json.load(f)
    with open(labels_path, 'r', encoding='utf-8') as f:
        labels_data = json.load(f)
    return docs_data, citations_data, labels_data

def normalize_paper_id(paper_id):
    """Removes author suffixes from paper IDs if present."""
    return paper_id.split('_')[0]

def extract_sentences(docs_data, citations_data, output_file):
    extracted_data = {}
    for paper_id, content in docs_data.items():
        normalized_id = normalize_paper_id(paper_id)
        sections = content.get("sections", {})
        
        # Extract Abstract and Conclusion sentences
        abstract_sentences = [s["text"] for s in sections.get("Abstract", {}).get("sentences", [])]
        conclusion_sentences = [s["text"] for s in sections.get("Conclusion", {}).get("sentences", [])]
        
        # Collect all sentences mapped by sid
        all_sentences = {}
        for sec_name, sec_content in sections.items():
            for sent in sec_content.get("sentences", []):
                sid = sent.get("sid")
                if sid is not None:
                    all_sentences[int(sid)] = sent.get("text", "")
        
        # Debug: Print available sids in the document
        print(f"Paper {paper_id} available sids: {list(all_sentences.keys())}")
        
        # Extract cited text spans using refer_sids
        cited_texts = []
        if normalized_id in citations_data:
            for citation in citations_data[normalized_id]:
                refer_sids = citation.get("refer_sids", [])
                print(f"Paper {paper_id} citation refer_sids (raw): {refer_sids}")
                
                # Ensure refer_sids are converted to integers
                refer_sids = [int(sid) for sid in refer_sids if isinstance(sid, (int, str)) and str(sid).isdigit()]
                print(f"Paper {paper_id} citation refer_sids (converted): {refer_sids}")
                
                for sid in refer_sids:
                    if sid in all_sentences:
                        cited_texts.append(all_sentences[sid])
                    else:
                        print(f"⚠️ Missing cited sid {sid} in paper {paper_id}")
        
        
        print(f"Paper {paper_id}: {len(cited_texts)} cited sentences found.")
        extracted_data[paper_id] = {
            "input_sentences": list(set(abstract_sentences + conclusion_sentences + cited_texts))
        }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(extracted_data, f, indent=4)
    print(f"✅ Extracted sentences saved to {output_file}")

def compute_rouge(reference_list, generated):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Compute ROUGE against multiple references
    scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    for reference in reference_list:
        score = scorer.score(reference, generated)
        for k in scores.keys():
            scores[k].append(score[k].fmeasure)
    
    # Return average ROUGE scores
    return {k: sum(v)/len(v) for k, v in scores.items()} if scores["rouge1"] else {}

def main():
    docs_path = "data/raw/test_docs_v5.json"
    citations_path = "data/raw/test_citations.json"
    labels_path = "data/raw/test_label.json"
    extracted_path = "data/processed/test_extracted.json"
    
    docs_data, citations_data, labels_data = load_test_data(docs_path, citations_path, labels_path)
    extract_sentences(docs_data, citations_data, extracted_path)
    
    with open(extracted_path, "r", encoding="utf-8") as f:
        extracted_data = json.load(f)
    
    results = []
    for paper_id, content in extracted_data.items():
        input_sentences = content.get("input_sentences", [])
        if not input_sentences:
            continue
        
        summary = summarize_paper(sentences=input_sentences)
        reference_summaries = labels_data.get(paper_id, [])
        
        rouge_scores = compute_rouge(reference_summaries, " ".join(summary))
        
        results.append({
            "paper_id": paper_id,
            "summarized_text": " ".join(summary),
            "rouge_score": rouge_scores
        })
    
    os.makedirs("test", exist_ok=True)

    with open("test/summary_results.json", "w", encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    
    print("Summarization complete. Results saved to summary_results.json")

if __name__ == "__main__":
    main()
