"""
Unit and integration tests for FertilizerRecommendationService and FertilizerAgent (Week 4).
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.services.fertilizer_service import (
    FertilizerRecommendationService,
    recommend_fertilizer,
)
from backend.agents.fertilizer_agent import (
    FertilizerAgent,
    create_recommend_fertilizer_tool,
)
from backend.schemas.fertilizer_schemas import FertilizerInputSchema


@pytest.fixture
def agent():
    """Fixture providing initialized FertilizerAgent."""
    return FertilizerAgent()


@pytest.fixture
def client():
    """Fixture providing FastAPI TestClient."""
    return TestClient(app)


def test_fertilizer_service_initialization():
    """Test ML service artifact loading."""
    service = FertilizerRecommendationService()
    assert service.model is not None
    assert service.scaler is not None
    assert service.soil_encoder is not None
    assert service.crop_encoder is not None


def test_recommend_fertilizer_urea_for_nitrogen_deficit():
    """Test that severe nitrogen deficiency triggers Urea recommendation."""
    result = recommend_fertilizer(
        nitrogen=12.0,
        phosphorus=55.0,
        potassium=50.0,
        soil_type="Loamy",
        crop_type="Rice"
    )
    assert result["recommended_fertilizer"] == "Urea"
    assert result["confidence"] > 0.70
    assert result["dosage_kg_per_acre"] > 0
    assert result["nutrient_report"]["nitrogen_status"] == "Deficient"


def test_recommend_fertilizer_dap_for_phosphorus_deficit():
    """Test that severe phosphorus deficiency triggers DAP recommendation."""
    result = recommend_fertilizer(
        nitrogen=60.0,
        phosphorus=10.0,
        potassium=55.0,
        soil_type="Black",
        crop_type="Wheat"
    )
    assert result["recommended_fertilizer"] in ["DAP", "14-35-14"]
    assert result["nutrient_report"]["phosphorus_status"] == "Deficient"


def test_fertilizer_tool_invocation():
    """Test direct invocation of the LangChain tool."""
    tool = create_recommend_fertilizer_tool()
    result = tool.invoke({
        "nitrogen": 18.0,
        "phosphorus": 55.0,
        "potassium": 50.0,
        "soil_type": "Loamy",
        "crop_type": "Rice"
    })
    assert isinstance(result, dict)
    assert result["recommended_fertilizer"] == "Urea"
    assert "dosage_kg_per_acre" in result


def test_fertilizer_agent_run_success(agent):
    """Test conversational agent with complete farmer query."""
    query = (
        "My soil has 18 nitrogen, 55 phosphorus, 50 potassium. "
        "I am planning to grow Rice in loamy soil. What fertilizer should I apply?"
    )
    response = agent.run(query)
    assert response.status == "success"
    assert response.recommended_fertilizer == "Urea"
    assert response.dosage_kg_per_acre is not None
    assert response.dosage_kg_per_acre > 0
    assert len(response.missing_parameters) == 0
    assert "Neem-Coated" in response.message or "Urea" in response.message


def test_fertilizer_agent_run_needs_clarification(agent):
    """Test conversational agent prompting for missing NPK data."""
    query = "What fertilizer is best for growing Rice in clayey soil?"
    response = agent.run(query)
    assert response.status == "needs_clarification"
    assert response.recommended_fertilizer is None
    assert "nitrogen" in response.missing_parameters
    assert "phosphorus" in response.missing_parameters
    assert "potassium" in response.missing_parameters


def test_api_predict_fertilizer_endpoint(client):
    """Test POST /predict/fertilizer endpoint."""
    payload = {
        "nitrogen": 18.0,
        "phosphorus": 55.0,
        "potassium": 50.0,
        "soil_type": "Loamy",
        "crop_type": "Rice"
    }
    response = client.post("/predict/fertilizer", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["recommended_fertilizer"] == "Urea"
    assert data["dosage_kg_per_acre"] > 0


def test_api_agent_fertilizer_endpoint(client):
    """Test POST /agent/fertilizer endpoint."""
    payload = {
        "query": "My soil has 18 nitrogen, 55 phosphorus, 50 potassium. Crop is Rice in loamy soil."
    }
    response = client.post("/agent/fertilizer", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["recommended_fertilizer"] == "Urea"
