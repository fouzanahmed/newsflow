"""Minimal TF-IDF retriever over a local sample news corpus.

Deliberately not FAISS/Chroma + embeddings: that would require either an
embeddings API key or a heavy local model dependency just to run tests.
TF-IDF + cosine similarity is genuine retrieval, fully local, and
deterministic -- swap in a vector store here if you want embeddings-based
retrieval; the interface (retrieve(query, k)) would stay the same.
"""
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = Path(__file__).parent / "data" / "sample_news.json"

REQUIRED_KEYS = ("id", "title", "text")


def validate_corpus(documents, source="corpus") -> None:
    """Raise ValueError with a readable message if the corpus is malformed.

    Without this, a bad corpus file surfaces as a bare KeyError or as
    sklearn's "empty vocabulary" error, neither of which says what is wrong
    with the file.
    """
    if not isinstance(documents, list):
        raise ValueError(
            f"{source}: expected a JSON list of documents, got {type(documents).__name__}"
        )
    if not documents:
        raise ValueError(f"{source}: corpus is empty; at least one document is required")

    seen_ids = {}
    for index, doc in enumerate(documents):
        if not isinstance(doc, dict):
            raise ValueError(
                f"{source}: document at index {index} must be an object, "
                f"got {type(doc).__name__}"
            )
        missing = [key for key in REQUIRED_KEYS if key not in doc]
        if missing:
            raise ValueError(
                f"{source}: document at index {index} is missing required "
                f"key(s): {', '.join(missing)}"
            )
        for key in REQUIRED_KEYS:
            value = doc[key]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"{source}: document at index {index} has an empty or "
                    f"non-string '{key}'"
                )
        doc_id = doc["id"]
        if doc_id in seen_ids:
            raise ValueError(
                f"{source}: duplicate document id '{doc_id}' at indexes "
                f"{seen_ids[doc_id]} and {index}"
            )
        seen_ids[doc_id] = index


class NewsRetriever:
    def __init__(self, data_path: Path = DATA_PATH):
        with open(data_path, "r", encoding="utf-8") as f:
            self.documents = json.load(f)
        validate_corpus(self.documents, source=str(data_path))
        corpus = [f"{d['title']} {d['text']}" for d in self.documents]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        try:
            self.doc_matrix = self.vectorizer.fit_transform(corpus)
        except ValueError as exc:
            # e.g. every title/text consists only of English stop words.
            raise ValueError(
                f"{data_path}: corpus has no indexable terms after stop-word "
                f"removal ({exc})"
            ) from exc

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.doc_matrix)[0]
        ranked = sorted(
            zip(self.documents, scores), key=lambda pair: pair[1], reverse=True
        )
        return [
            {**doc, "score": float(score)} for doc, score in ranked[:k] if score > 0
        ]
