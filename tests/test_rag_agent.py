"""
Unit and integration tests for Agricultural Knowledge RAG Agent (Week 6).
Covers document loader, FAISS vector store, LangChain RAG tool, and FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.rag.document_loader import AgriculturalDocumentLoader
from backend.rag.vector_store import AgriculturalVectorStore, get_vector_store
from backend.agents.knowledge_agent import KnowledgeAgent, create_knowledge_tool


@pytest.fixture
def client():
    """Fixture providing FastAPI TestClient."""
    return TestClient(app)


def test_document_loader():
    """Test loading and chunking of markdown guidelines."""
    loader = AgriculturalDocumentLoader()
    chunks = loader.load_guidelines()
    assert len(chunks) >= 4
    section_titles = [c.section for c in chunks]
    assert any("Rice" in s for s in section_titles)
    assert any("Cotton" in s for s in section_titles)


def test_vector_store_indexing_and_search():
    """Test FAISS index building and semantic retrieval."""
    store = AgriculturalVectorStore()
    num_indexed = store.build_index()
    assert num_indexed > 0

    results = store.search("blast disease in paddy", top_k=2)
    assert len(results) > 0
    top_result = results[0]
    assert "Rice" in top_result["section"] or "blast" in top_result["content"].lower()
    assert top_result["score"] > 0.1


def test_knowledge_tool_invocation():
    """Test LangChain RAG tool execution."""
    tool = create_knowledge_tool()
    results = tool.invoke({"query": "How to control pink bollworm in cotton?"})
    assert isinstance(results, list)
    assert len(results) > 0
    assert any("Cotton" in r["section"] for r in results)


def test_knowledge_agent_run():
    """Test end-to-end KnowledgeAgent answer synthesis."""
    agent = KnowledgeAgent()
    response = agent.run("What fungicide is recommended for yellow rust in wheat?")
    assert len(response.citations) > 0
    assert "Propiconazole" in response.answer or "Tilt" in response.answer or "Wheat" in response.answer
    assert response.citations[0].source == "sample_icar_guidelines.md"


def test_api_knowledge_endpoint(client):
    """Test POST /agent/knowledge FastAPI endpoint."""
    payload = {"query": "How do I control fall armyworm in maize?"}
    response = client.post("/agent/knowledge", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert len(data["citations"]) > 0
    assert "Maize" in data["citations"][0]["section"] or "armyworm" in data["raw_context"].lower()
