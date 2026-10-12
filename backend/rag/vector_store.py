"""
FAISS Vector Store for Agricultural Advisory Retrieval (Week 6).
Indexes agronomic guidelines using dense vector projections and cosine similarity.
"""

from typing import List, Dict, Any, Optional
import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from backend.rag.document_loader import AgriculturalDocumentLoader, DocumentChunk


class AgriculturalVectorStore:
    """FAISS-indexed in-memory vector store for ICAR/KVK agronomic documents."""

    def __init__(self, top_features: int = 256):
        """Initialize vector store and embedding engine."""
        self.top_features = top_features
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=self.top_features,
            sublinear_tf=True
        )
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks: List[DocumentChunk] = []

    def build_index(self, chunks: Optional[List[DocumentChunk]] = None) -> int:
        """
        Build FAISS vector index from document chunks.
        
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
        faiss.normalize_L2(tfidf_matrix)

        dim = tfidf_matrix.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(tfidf_matrix)

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
        faiss.normalize_L2(q_vec)

        k = min(top_k, len(self.chunks))
        scores, indices = self.index.search(q_vec, k)

        results: List[Dict[str, Any]] = []
        for i in range(k):
            idx = int(indices[0][i])
            score = float(scores[0][i])
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
