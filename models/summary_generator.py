import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def is_redundant(sentence, summary_sentences, threshold=0.5):
    if not summary_sentences:
        return False
    vectorizer = TfidfVectorizer().fit_transform([sentence] + summary_sentences)
    similarity_matrix = cosine_similarity(vectorizer)
    max_similarity = np.max(similarity_matrix[0, 1:])
    return max_similarity > threshold

def hybrid_1(sentences, salience_scores, length_limit):
    """
    Extractive summarization of (abstract ∪ cited text spans).
    :param sentences: List of sentences in I (abstract + cited text spans)
    :param salience_scores: Corresponding salience scores
    :param length_limit: Maximum number of words allowed in the summary
    :return: Extracted summary
    """
    sorted_sentences = [s for _, s in sorted(zip(salience_scores, sentences), reverse=True)]
    summary = []
    total_length = 0
    
    for sentence in sorted_sentences:
        if len(sentence.split()) > 8 and not is_redundant(sentence, summary):
            if total_length + len(sentence.split()) <= length_limit:
                summary.append(sentence)
                total_length += len(sentence.split())
            else:
                break
    
    return sorted(summary, key=lambda x: sentences.index(x))  # Restore original order

def hybrid_2(abstract, cited_text_spans, salience_scores, length_limit):
    """
    Augmentation of abstract with salient cited text spans.
    :param abstract: The original abstract
    :param cited_text_spans: List of cited text spans from I
    :param salience_scores: Corresponding salience scores for cited text spans
    :param length_limit: Maximum number of words allowed in the summary
    :return: Augmented summary
    """
    sorted_cited_spans = [s for _, s in sorted(zip(salience_scores, cited_text_spans), reverse=True)]
    summary = abstract[:]
    total_length = sum(len(s.split()) for s in summary)
    
    for sentence in sorted_cited_spans:
        if len(sentence.split()) > 8 and not is_redundant(sentence, summary):
            if total_length + len(sentence.split()) <= length_limit:
                summary.append(sentence)
                total_length += len(sentence.split())
            else:
                break
    
    return sorted(summary, key=lambda x: (abstract + cited_text_spans).index(x))  # Restore original order
