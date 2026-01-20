from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict


def extract_keywords(
    segments: List[Dict],
    top_k: int = 5
) -> List[str]:
    """
    Extract top keywords for a topic using TF-IDF.
    """

    texts = [s["text"] for s in segments]

    if not texts:
        return []

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=1000
    )

    tfidf_matrix = vectorizer.fit_transform(texts)
    scores = tfidf_matrix.sum(axis=0).A1
    vocab = vectorizer.get_feature_names_out()

    ranked = sorted(
        zip(vocab, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [word for word, _ in ranked[:top_k]]
