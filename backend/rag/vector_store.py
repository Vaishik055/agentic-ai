"""
FAISS Vector Store for Agricultural Advisory Retrieval (Week 6).
Indexes agronomic guidelines using dense vector projections and cosine similarity.
Includes pure-NumPy fallback for environments (like Python 3.14) where FAISS binaries are not available.
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from backend.rag.document_loader import AgriculturalDocumentLoader, DocumentChunk

logger = logging.getLogger(__name__)

try:
    import faiss
except ImportError:
    faiss = None
    logger.info("FAISS not found in environment. Using pure-NumPy cosine similarity vector engine.")


class AgriculturalVectorStore:
    """FAISS-indexed in-memory vector store for ICAR/KVK agronomic documents with NumPy fallback."""

    def __init__(self, top_features: int = 1024):
        """Initialize vector store and embedding engine."""
        self.top_features = top_features
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=self.top_features,
            sublinear_tf=True
        )
        self.index: Any = None
        self.matrix: Optional[np.ndarray] = None
        self.chunks: List[DocumentChunk] = []

    def build_index(self, chunks: Optional[List[DocumentChunk]] = None) -> int:
        """
        Build FAISS vector index (or NumPy dense index) from document chunks.
        
        Args:
            chunks: List of DocumentChunk objects. If None, loaded automatically from docs/.
            
        Returns:
            Number of indexed documents.
        """
        if chunks is None:
            loader = AgriculturalDocumentLoader()
            chunks = loader.load_guidelines()

        if not chunks:
            # Fallback guideline chunk if no external documents exist
            chunks = [
                DocumentChunk(
                    content="Section: Rice Management\nSpray Tricyclazole 75 WP at 0.6 g/L for blast control. Maintain 3-5 cm water during tillering.",
                    source="fallback_guidelines.md",
                    section="Rice Management"
                )
            ]

        self.chunks = chunks
        corpus = [c.content for c in self.chunks]

        # Generate TF-IDF embeddings matrix
        tfidf_matrix = self.vectorizer.fit_transform(corpus).toarray().astype("float32")

        if faiss is not None:
            faiss.normalize_L2(tfidf_matrix)
            dim = tfidf_matrix.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(tfidf_matrix)
        else:
            # Pure NumPy cosine normalization (Python 3.14 compatible)
            norms = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)
            self.matrix = np.divide(tfidf_matrix, np.maximum(norms, 1e-9), out=tfidf_matrix, where=norms != 0)
            self.index = True

        return len(self.chunks)

    def search(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """
        Search vector database for top matching agricultural practices.
        
        Args:
            query: Natural language farmer question.
            top_k: Number of most relevant sections to retrieve.
            
        Returns:
            List of matching records with score, content, source, and section.
        """
        if self.index is None or not self.chunks:
            self.build_index()

        q_vec = self.vectorizer.transform([query]).toarray().astype("float32")
        k = min(top_k, len(self.chunks))

        if faiss is not None and hasattr(self.index, "search"):
            faiss.normalize_L2(q_vec)
            scores, indices = self.index.search(q_vec, k)
            top_indices = [int(idx) for idx in indices[0]]
            top_scores = [float(s) for s in scores[0]]
        else:
            q_norm = np.linalg.norm(q_vec)
            if q_norm > 0:
                q_vec = q_vec / q_norm
            similarities = np.dot(self.matrix, q_vec[0])
            sorted_idx = np.argsort(similarities)[::-1][:k]
            top_indices = [int(idx) for idx in sorted_idx]
            top_scores = [float(similarities[idx]) for idx in top_indices]

        results: List[Dict[str, Any]] = []
        for i in range(k):
            idx = top_indices[i]
            score = top_scores[i]
            chunk = self.chunks[idx]
            results.append({
                "score": round(score, 4),
                "section": chunk.section,
                "source": chunk.source,
                "content": chunk.content
            })

        return results


# Global singleton instance for efficient sharing across agents
_vector_store_instance = None


def get_vector_store() -> AgriculturalVectorStore:
    """Get or initialize singleton instance of AgriculturalVectorStore."""
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = AgriculturalVectorStore()
        _vector_store_instance.build_index()
    return _vector_store_instance