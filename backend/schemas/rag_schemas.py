"""
Pydantic schemas for Agricultural Knowledge RAG Agent (Week 6).
Defines contracts for agronomic retrieval queries and citation-backed responses.
"""

from typing import List, Dict, Any
from pydantic import BaseModel, Field


class KnowledgeQueryRequest(BaseModel):
    """Input payload for Agricultural Knowledge RAG query."""
    query: str = Field(
        ...,
        description="Agronomic question (pest control, disease diagnosis, seed treatment, sowing calendar).",
        examples=["How do I manage pink bollworm in cotton?"]
    )


class KnowledgeSourceItem(BaseModel):
    """Citation metadata for retrieved guideline chunk."""
    section: str = Field(..., description="Document section heading.")
    source: str = Field(..., description="Source filename or manual title.")
    score: float = Field(..., description="Vector similarity relevance score.")


class KnowledgeAgentResponse(BaseModel):
    """Structured response from the Agricultural Knowledge RAG Agent."""
    query: str = Field(..., description="Original farmer query.")
    answer: str = Field(..., description="Synthesized agronomic recommendation.")
    citations: List[KnowledgeSourceItem] = Field(..., description="Supporting guidelines retrieved from ICAR/KVK corpus.")
    raw_context: str = Field(..., description="Extracted context excerpts.")
