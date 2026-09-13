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


class NewsRetriever:
    def __init__(self, data_path: Path = DATA_PATH):
        with open(data_path, "r", encoding="utf-8") as f:
            self.documents = json.load(f)
        corpus = [f"{d['title']} {d['text']}" for d in self.documents]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.doc_matrix = self.vectorizer.fit_transform(corpus)

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.doc_matrix)[0]
        ranked = sorted(
            zip(self.documents, scores), key=lambda pair: pair[1], reverse=True
        )
        return [
            {**doc, "score": float(score)} for doc, score in ranked[:k] if score > 0
        ]
