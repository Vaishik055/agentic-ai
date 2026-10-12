"""
Pydantic schemas for External Data Gathering Agents (Week 5).
Defines contracts for Weather, Soil Profile, and Mandi Market Price Agents.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


# ============================================================================
# Weather Schemas
# ============================================================================

class WeatherAgentRequest(BaseModel):
    """Input payload for Weather Agent."""
    location: str = Field(
        ...,
        description="Location name, district, or city (e.g. 'Hyderabad', 'Guntur', 'Pune').",
        examples=["Hyderabad, Telangana"]
    )


class WeatherDataResponse(BaseModel):
    """Structured weather data returned by Weather Agent."""
    location: str = Field(..., description="Normalized location name.")
    temperature: float = Field(..., description="Ambient temperature in °C.")
    humidity: float = Field(..., description="Relative air humidity in %.")
    rainfall_forecast_mm: float = Field(..., description="Estimated seasonal or 7-day rainfall in mm.")
    weather_condition: str = Field(..., description="Brief weather description (e.g. 'Sunny', 'Rainy').")
    source: str = Field(..., description="Data source ('OpenWeatherMap API' or 'Agro-Climatic Database').")
    message: str = Field(..., description="Farmer-friendly summary of climatic conditions.")


# ============================================================================
# Soil Profile Schemas
# ============================================================================

class SoilAgentRequest(BaseModel):
    """Input payload for Soil Agent."""
    location: str = Field(
        ...,
        description="District, state, or agro-climatic region (e.g. 'Nizamabad', 'Punjab').",
        examples=["Nizamabad, Telangana"]
    )


class SoilProfileResponse(BaseModel):
    """Regional soil profile data returned by Soil Agent."""
    region: str = Field(..., description="Identified region or district.")
    dominant_soil_type: str = Field(..., description="Dominant soil classification (Black, Red, Loamy, Alluvial, etc.).")
    average_ph: float = Field(..., description="Typical regional soil pH.")
    average_nitrogen: float = Field(..., description="Estimated regional Nitrogen level (kg/ha).")
    average_phosphorus: float = Field(..., description="Estimated regional Phosphorus level (kg/ha).")
    average_potassium: float = Field(..., description="Estimated regional Potassium level (kg/ha).")
    organic_carbon_percent: float = Field(..., description="Soil organic carbon percentage.")
    source: str = Field(..., description="Data source ('SoilGrids API' or 'ICAR Soil Atlas').")
    message: str = Field(..., description="Agronomic summary of the regional soil.")


# ============================================================================
# Mandi Market Price Schemas
# ============================================================================

class MarketAgentRequest(BaseModel):
    """Input payload for Market Price Agent."""
    crop: str = Field(
        ...,
        description="Crop commodity name (e.g. 'Rice', 'Cotton', 'Maize').",
        examples=["Rice"]
    )
    state: Optional[str] = Field(
        "Telangana",
        description="Indian state for regional mandi pricing.",
        examples=["Telangana"]
    )


class MarketPriceResponse(BaseModel):
    """Mandi price details returned by Market Price Agent."""
    commodity: str = Field(..., description="Crop commodity name.")
    state: str = Field(..., description="State name.")
    market_mandi: str = Field(..., description="Name of the reporting mandi wholesale market.")
    min_price_per_quintal: float = Field(..., description="Minimum wholesale price in INR/quintal.")
    modal_price_per_quintal: float = Field(..., description="Modal (most common) market price in INR/quintal.")
    max_price_per_quintal: float = Field(..., description="Maximum wholesale price in INR/quintal.")
    unit: str = Field("INR / Quintal (100 kg)", description="Pricing currency and unit.")
    source: str = Field(..., description="Data source ('Agmarknet API' or 'Mandi Trade Index').")
    message: str = Field(..., description="Economic guidance on crop value.")
