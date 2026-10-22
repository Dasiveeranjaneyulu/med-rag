
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from loguru import logger
import pickle
import os
from backend.app.core.config import settings

class BM25Retriever:
    """BM25-style retriever using TF-IDF + cosine similarity as a simple approximation."""
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        self.documents = []
        self.tfidf_matrix = None
        self.pickle_path = os.path.join(str(settings.VECTORSTORE_DIR), "bm25_index.pkl")
        self._load_index()

    def _save_index(self):
        """Save BM25 index to disk."""
        os.makedirs(str(settings.VECTORSTORE_DIR), exist_ok=True)
        with open(self.pickle_path, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'vectorizer': self.vectorizer,
                'tfidf_matrix': self.tfidf_matrix
            }, f)
        logger.info(f"BM25 index saved to {self.pickle_path}")

    def _load_index(self):
        """Load BM25 index from disk if exists."""
        if os.path.exists(self.pickle_path):
            try:
                with open(self.pickle_path, 'rb') as f:
                    data = pickle.load(f)
                    self.documents = data['documents']
                    self.vectorizer = data['vectorizer']
                    self.tfidf_matrix = data['tfidf_matrix']
                logger.info(f"BM25 index loaded from {self.pickle_path} with {len(self.documents)} docs")
            except Exception as e:
                logger.error(f"Failed to load BM25 index: {e}")

    def rebuild(self, documents: list[str]):
        """Replace the BM25 index with the given documents."""
        self.documents = list(documents)
        if self.documents:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
            self._save_index()
        else:
            self.tfidf_matrix = None
        logger.info(f"BM25 index rebuilt with {len(self.documents)} docs")

    def add_documents(self, documents):
        """Append documents to BM25 index."""
        self.documents.extend(documents)
        if len(self.documents) > 0:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
            self._save_index()
        logger.info(f"BM25 index now has {len(self.documents)} docs")

    def query(self, query_text, n_results=5):
        """Query BM25 index."""
        if not self.documents or self.tfidf_matrix is None:
            return [], []
        
        query_vec = self.vectorizer.transform([query_text])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        top_indices = np.argsort(similarities)[::-1][:n_results]
        
        top_docs = [self.documents[i] for i in top_indices]
        top_scores = [float(similarities[i]) for i in top_indices]
        
        return top_docs, top_scores
