"""
Unit and integration tests for Phase 5 Evaluation Agents (Week 7).
Covers SustainabilityAgent, YieldProfitAgent, RotationPlannerAgent, and FastAPI routes.
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.agents.sustainability_agent import SustainabilityAgent, create_sustainability_tool
from backend.agents.yield_profit_agent import YieldProfitAgent, create_profitability_tool
from backend.agents.rotation_planner_agent import RotationPlannerAgent, create_rotation_tool
from backend.schemas.evaluation_schemas import (
    SustainabilityRequest,
    ProfitabilityRequest,
    RotationPlanRequest,
)


@pytest.fixture
def client():
    """Fixture providing FastAPI TestClient."""
    return TestClient(app)


# ============================================================================
# Sustainability Agent Tests
# ============================================================================

def test_sustainability_drip_vs_flood():
    """Test that drip irrigation and organic amendments achieve higher sustainability scores."""
    agent = SustainabilityAgent()

    high_eco = agent.evaluate(SustainabilityRequest(
        crop="Rice",
        irrigation_method="Drip",
        fertilizer_used="Neem-Coated Urea",
        soil_organic_carbon=0.85,
        rainfall_mm=220.0
    ))

    low_eco = agent.evaluate(SustainabilityRequest(
        crop="Rice",
        irrigation_method="Flood",
        fertilizer_used="Urea",
        soil_organic_carbon=0.35,
        rainfall_mm=220.0
    ))

    assert high_eco.sustainability_index > low_eco.sustainability_index
    assert high_eco.sustainability_index >= 75.0
    assert "Eco-Leader" in high_eco.sustainability_grade or "Sustainable" in high_eco.sustainability_grade


def test_sustainability_tool_invocation():
    """Test LangChain sustainability tool execution."""
    tool = create_sustainability_tool()
    res = tool.invoke({
        "crop": "Cotton",
        "irrigation_method": "Drip",
        "fertilizer_used": "Neem-Coated Urea",
        "soil_organic_carbon": 0.60,
        "rainfall_mm": 180.0
    })
    assert "sustainability_index" in res
    assert res["sustainability_index"] > 50.0


# ============================================================================
# Yield & Profitability Agent Tests
# ============================================================================

def test_yield_profitability_calculation():
    """Test yield forecasting and financial ROI computation."""
    agent = YieldProfitAgent()
    res = agent.evaluate(ProfitabilityRequest(
        crop="Cotton",
        land_area_acres=2.0,
        mandi_price_per_quintal=7120.0
    ))
    # Cotton benchmark: 10 q/acre, ₹28,000 cost/acre
    assert res.expected_yield_quintals == 20.0
    assert res.estimated_cost_inr == 56000.0
    assert res.gross_revenue_inr == 20.0 * 7120.0
    assert res.net_profit_inr == (20.0 * 7120.0) - 56000.0
    assert res.roi_percentage > 100.0


def test_profitability_tool_invocation():
    """Test LangChain profitability tool execution."""
    tool = create_profitability_tool()
    res = tool.invoke({
        "crop": "Rice",
        "land_area_acres": 3.0,
        "mandi_price_per_quintal": 2350.0
    })
    assert "net_profit_inr" in res
    assert res["expected_yield_quintals"] == 66.0  # 22 q/acre * 3


# ============================================================================
# Crop Rotation Planner Agent Tests
# ============================================================================

def test_crop_rotation_plan_sequence():
    """Test 3-season rotation schedule and biological nitrogen gain."""
    agent = RotationPlannerAgent()
    plan = agent.plan_rotation(RotationPlanRequest(
        current_crop="Rice",
        soil_type="Black",
        irrigation_availability="Assured"
    ))
    assert len(plan.annual_sequence) == 3
    seasons = [s.season for s in plan.annual_sequence]
    assert any("Kharif" in s for s in seasons)
    assert any("Rabi" in s for s in seasons)
    assert any("Zaid" in s for s in seasons)
    assert plan.nitrogen_gain_kg_ha >= 40.0


def test_rotation_tool_invocation():
    """Test LangChain rotation tool execution."""
    tool = create_rotation_tool()
    res = tool.invoke({
        "current_crop": "Cotton",
        "soil_type": "Black",
        "irrigation_availability": "Assured"
    })
    assert "annual_sequence" in res
    assert "nitrogen_gain_kg_ha" in res


# ============================================================================
# FastAPI Integration Tests
# ============================================================================

def test_api_sustainability_endpoint(client):
    """Test POST /agent/sustainability endpoint."""
    res = client.post("/agent/sustainability", json={
        "crop": "Maize",
        "irrigation_method": "Drip",
        "fertilizer_used": "Neem-Coated Urea",
        "soil_organic_carbon": 0.65,
        "rainfall_mm": 180.0
    })
    assert res.status_code == 200
    data = res.json()
    assert "sustainability_index" in data
    assert data["sustainability_index"] > 60.0


def test_api_profitability_endpoint(client):
    """Test POST /agent/profitability endpoint."""
    res = client.post("/agent/profitability", json={
        "crop": "Rice",
        "land_area_acres": 2.0,
        "mandi_price_per_quintal": 2350.0
    })
    assert res.status_code == 200
    data = res.json()
    assert data["expected_yield_quintals"] == 44.0
    assert data["net_profit_inr"] > 0


def test_api_rotation_endpoint(client):
    """Test POST /agent/rotation endpoint."""
    res = client.post("/agent/rotation", json={
        "current_crop": "Rice",
        "soil_type": "Black",
        "irrigation_availability": "Assured"
    })
    assert res.status_code == 200
    data = res.json()
    assert len(data["annual_sequence"]) == 3
    assert data["nitrogen_gain_kg_ha"] > 0
