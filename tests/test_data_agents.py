"""
Unit and integration tests for External Data Gathering Agents (Week 5).
Covers WeatherAgent, SoilAgent, MarketAgent, and their FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.agents.weather_agent import WeatherAgent, create_weather_tool
from backend.agents.soil_agent import SoilAgent, create_soil_tool
from backend.agents.market_agent import MarketAgent, create_market_tool
from backend.utils.api_clients import WeatherClient, SoilClient, MarketPriceClient


@pytest.fixture
def client():
    """Fixture providing FastAPI TestClient."""
    return TestClient(app)


# ============================================================================
# Weather Agent Tests
# ============================================================================

def test_weather_client_known_location():
    """Test weather retrieval for mapped agro-climatic region."""
    client = WeatherClient()
    data = client.fetch_weather("Nizamabad")
    assert "Nizamabad" in data["location"]
    assert 20.0 <= data["temperature"] <= 45.0
    assert 0.0 <= data["humidity"] <= 100.0
    assert data["rainfall_forecast_mm"] > 0


def test_weather_tool_invocation():
    """Test LangChain tool execution for weather."""
    tool = create_weather_tool()
    result = tool.invoke({"location": "Hyderabad"})
    assert isinstance(result, dict)
    assert "temperature" in result
    assert "humidity" in result


def test_weather_agent_run():
    """Test conversational WeatherAgent with query parsing."""
    agent = WeatherAgent()
    resp = agent.run("What is the weather in Nizamabad?")
    assert "Nizamabad" in resp.location
    assert resp.temperature > 0
    assert "Rainfall" in resp.message


# ============================================================================
# Soil Agent Tests
# ============================================================================

def test_soil_client_known_region():
    """Test regional soil profile lookup."""
    client = SoilClient()
    profile = client.fetch_soil_profile("Nizamabad")
    assert profile["dominant_soil_type"] == "Black"
    assert 5.0 <= profile["average_ph"] <= 8.5
    assert profile["average_nitrogen"] > 0


def test_soil_tool_invocation():
    """Test LangChain tool execution for soil profile."""
    tool = create_soil_tool()
    result = tool.invoke({"location": "Guntur"})
    assert "dominant_soil_type" in result
    assert "average_ph" in result


def test_soil_agent_run():
    """Test conversational SoilAgent with text query."""
    agent = SoilAgent()
    resp = agent.run("Tell me about the soil profile around Nizamabad.")
    assert "Nizamabad" in resp.region
    assert resp.dominant_soil_type == "Black"
    assert resp.average_ph > 0


# ============================================================================
# Market Agent Tests
# ============================================================================

def test_market_client_known_crop():
    """Test mandi price lookup for commodity."""
    client = MarketPriceClient()
    data = client.fetch_market_price(crop="Rice", state="Telangana")
    assert data["commodity"] == "Rice"
    assert data["modal_price_per_quintal"] >= 2000.0
    assert data["max_price_per_quintal"] >= data["min_price_per_quintal"]


def test_market_tool_invocation():
    """Test LangChain tool execution for mandi prices."""
    tool = create_market_tool()
    result = tool.invoke({"crop": "Cotton", "state": "Telangana"})
    assert "modal_price_per_quintal" in result
    assert result["modal_price_per_quintal"] > 5000.0


def test_market_agent_run():
    """Test conversational MarketAgent with text query."""
    agent = MarketAgent()
    resp = agent.run("What is the market rate for Cotton in Telangana today?")
    assert resp.commodity == "Cotton"
    assert resp.modal_price_per_quintal > 5000.0
    assert "Adilabad" in resp.market_mandi or "Mandi" in resp.market_mandi


# ============================================================================
# FastAPI Endpoint Integration Tests
# ============================================================================

def test_api_weather_endpoint(client):
    """Test POST /agent/weather endpoint."""
    res = client.post("/agent/weather", json={"location": "Hyderabad"})
    assert res.status_code == 200
    data = res.json()
    assert "temperature" in data
    assert "humidity" in data


def test_api_soil_endpoint(client):
    """Test POST /agent/soil endpoint."""
    res = client.post("/agent/soil", json={"location": "Nizamabad"})
    assert res.status_code == 200
    data = res.json()
    assert data["dominant_soil_type"] == "Black"


def test_api_market_endpoint(client):
    """Test POST /agent/market endpoint."""
    res = client.post("/agent/market", json={"crop": "Rice", "state": "Telangana"})
    assert res.status_code == 200
    data = res.json()
    assert data["modal_price_per_quintal"] > 0
