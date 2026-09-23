"""
Unit and integration tests for CropRecommendationAgent and its FastAPI endpoints (Week 3).
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.agents.crop_recommendation_agent import (
    CropRecommendationAgent,
    create_predict_crop_tool
)


@pytest.fixture
def agent():
    """Fixture providing initialized CropRecommendationAgent."""
    return CropRecommendationAgent()


@pytest.fixture
def client():
    """Fixture providing FastAPI TestClient."""
    return TestClient(app)


def test_predict_crop_tool():
    """Test direct invocation of the LangChain tool."""
    tool = create_predict_crop_tool()
    result = tool.invoke({
        "nitrogen": 90.0,
        "phosphorus": 42.0,
        "potassium": 43.0,
        "temperature": 25.0,
        "humidity": 80.0,
        "ph": 6.5,
        "rainfall": 200.0
    })
    assert isinstance(result, dict)
    assert result["recommended_crop"] == "rice"
    assert result["confidence"] > 0.85


def test_agent_extract_full_parameters(agent):
    """Test extracting all 7 parameters from natural farmer sentences."""
    query = (
        "My soil has 90 nitrogen, 42 phosphorus, 43 potassium, "
        "temperature is 25°C, humidity 80%, pH 6.5, rainfall 200mm. What should I grow?"
    )
    extracted = agent.extract_parameters_regex(query)
    assert extracted["nitrogen"] == 90.0
    assert extracted["phosphorus"] == 42.0
    assert extracted["potassium"] == 43.0
    assert extracted["temperature"] == 25.0
    assert extracted["humidity"] == 80.0
    assert extracted["ph"] == 6.5
    assert extracted["rainfall"] == 200.0


def test_agent_extract_partial_parameters(agent):
    """Test identification of missing parameters."""
    query = "Soil has nitrogen 80 and pH 6.8. Recommend a crop."
    extracted = agent.extract_parameters_regex(query)
    missing = agent.identify_missing(extracted)
    assert extracted["nitrogen"] == 80.0
    assert extracted["ph"] == 6.8
    assert "phosphorus" in missing
    assert "rainfall" in missing
    assert "temperature" in missing


def test_agent_run_success(agent):
    """Test agent complete execution on valid query."""
    query = (
        "Farmer query: 90 nitrogen, 42 phosphorus, 43 potassium, "
        "temp 25 C, humidity 80%, ph 6.5, rain 200 mm. Please advise."
    )
    response = agent.run(query)
    assert response.status == "success"
    assert response.recommended_crop == "rice"
    assert response.confidence is not None
    assert response.confidence > 0.8
    assert len(response.missing_parameters) == 0
    assert "Rice" in response.message


def test_agent_run_needs_clarification(agent):
    """Test agent asking clarifying questions when required features are missing."""
    query = "I have 50 nitrogen and 30 phosphorus in my soil. What can I plant?"
    response = agent.run(query)
    assert response.status == "needs_clarification"
    assert response.recommended_crop is None
    assert len(response.missing_parameters) > 0
    assert "potassium" in response.missing_parameters
    assert "rainfall" in response.missing_parameters
    assert "still need" in response.message.lower()


def test_api_agent_crop_endpoint_success(client):
    """Test POST /agent/crop endpoint with complete parameters."""
    payload = {
        "query": "My soil has 90 nitrogen, 42 phosphorus, 43 potassium, temperature is 25°C, humidity 80%, pH 6.5, rainfall 200mm."
    }
    response = client.post("/agent/crop", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["recommended_crop"] == "rice"
    assert data["confidence"] > 0.85


def test_api_agent_crop_endpoint_clarification(client):
    """Test POST /agent/crop endpoint with incomplete parameters."""
    payload = {
        "query": "I only know that my soil has 80 nitrogen and pH 6.5."
    }
    response = client.post("/agent/crop", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "needs_clarification"
    assert data["recommended_crop"] is None
    assert "rainfall" in data["missing_parameters"]
