"""
Integration tests for Master Multi-Agent Orchestrator and Reasoning Engine (Week 8).
Validates end-to-end LangGraph state transitions, agent handoffs, and FastAPI orchestration routes.
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.agents.orchestrator import get_farm_orchestrator
from backend.schemas.orchestrator_schemas import OrchestratorQueryRequest


@pytest.fixture
def client():
    """Fixture providing FastAPI TestClient."""
    return TestClient(app)


def test_orchestrator_graph_compilation():
    """Test that LangGraph compiles and initializes cleanly."""
    orchestrator = get_farm_orchestrator()
    assert orchestrator.graph is not None


def test_orchestrator_end_to_end_full_features():
    """
    Test major project final deliverable:
    Farmer provides full soil and climate values -> All agents execute in sequence.
    """
    orchestrator = get_farm_orchestrator()
    req = OrchestratorQueryRequest(
        query="My soil has 90 nitrogen, 42 phosphorus, 43 potassium, temperature is 25°C, humidity 80%, pH 6.5, rainfall 200mm. What should I grow?",
        location="Hyderabad",
        land_area_acres=2.0,
        irrigation_method="Flood"
    )
    res = orchestrator.process_query(req)

    assert res.status == "success"
    # 1. Primary Recommendation
    assert res.primary_recommendation["crop"]["recommended_crop"] == "rice"
    assert res.primary_recommendation["crop"]["confidence"] > 0.85
    assert "recommended_fertilizer" in res.primary_recommendation["fertilizer"]

    # 2. Environmental Context
    assert "temperature" in res.environmental_context["weather"]
    assert "dominant_soil_type" in res.environmental_context["soil"]

    # 3. Market & Profitability
    assert res.market_and_profitability["profitability"]["expected_yield_quintals"] > 0
    assert res.market_and_profitability["profitability"]["net_profit_inr"] > 0
    assert res.market_and_profitability["profitability"]["roi_percentage"] > 0

    # 4. Sustainability & Rotation
    assert res.sustainability_and_planning["sustainability"]["sustainability_index"] > 50.0
    assert len(res.sustainability_and_planning["rotation"]["annual_sequence"]) == 3

    # 5. Explainable Reasoning
    assert "Rice" in res.explainable_reasoning
    assert "Net Profit" in res.explainable_reasoning or "Profit" in res.explainable_reasoning


def test_orchestrator_autonomous_enrichment():
    """
    Test Phase 3 dynamic data enrichment:
    Farmer provides location and partial soil -> Weather and Soil agents supply missing values.
    """
    orchestrator = get_farm_orchestrator()
    req = OrchestratorQueryRequest(
        query="I am a farmer in Nizamabad. What is the best crop for 3 acres of my farm?",
        location="Nizamabad",
        land_area_acres=3.0
    )
    res = orchestrator.process_query(req)

    assert res.status == "success"
    assert res.primary_recommendation["crop"]["recommended_crop"] is not None
    assert "Nizamabad" in res.environmental_context["weather"]["location"]
    assert res.market_and_profitability["profitability"]["land_area_acres"] == 3.0


def test_api_orchestrate_endpoint(client):
    """Test POST /agent/orchestrate endpoint."""
    payload = {
        "query": "My soil has 90 nitrogen, 42 phosphorus, 43 potassium, temperature is 25°C, humidity 80%, pH 6.5, rainfall 200mm. What should I grow?",
        "location": "Hyderabad, Telangana",
        "land_area_acres": 2.0
    }
    response = client.post("/agent/orchestrate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["primary_recommendation"]["crop"]["recommended_crop"] == "rice"
    assert "explainable_reasoning" in data
    assert len(data["explainable_reasoning"]) > 100
