"""
Pydantic schemas for Phase 5 Evaluation, Sustainability, and Farm Planning Agents (Week 7).
Defines contracts for Sustainability Index, Yield/Profit Forecasting, and Crop Rotation Plans.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


# ============================================================================
# Sustainability Scoring Schemas
# ============================================================================

class SustainabilityRequest(BaseModel):
    """Input parameters to assess farming sustainability."""
    crop: str = Field(..., description="Target crop (e.g. 'Rice', 'Maize', 'Cotton').", examples=["Rice"])
    irrigation_method: str = Field(
        "Flood",
        description="Irrigation technique: 'Drip', 'Sprinkler', 'Flood', 'Rainfed'.",
        examples=["Flood"]
    )
    fertilizer_used: str = Field(
        "Urea",
        description="Primary fertilizer used (e.g. 'Urea', 'Neem-Coated Urea', 'Organic Manure', 'DAP').",
        examples=["Neem-Coated Urea"]
    )
    soil_organic_carbon: float = Field(
        0.55,
        ge=0.0,
        le=5.0,
        description="Soil Organic Carbon percentage (0.0 to 5.0%).",
        examples=[0.55]
    )
    rainfall_mm: float = Field(
        200.0,
        ge=0.0,
        le=1500.0,
        description="Seasonal rainfall in mm.",
        examples=[200.0]
    )


class SustainabilityScoreBreakdown(BaseModel):
    """Component scores out of 25 for each sustainability pillar."""
    water_efficiency_score: float = Field(..., description="Score for water conservation (0-25).")
    soil_health_score: float = Field(..., description="Score for soil preservation and microbial carbon (0-25).")
    chemical_load_score: float = Field(..., description="Score for synthetic fertilizer/pesticide reduction (0-25).")
    carbon_climate_score: float = Field(..., description="Score for greenhouse gas mitigation (0-25).")


class SustainabilityResponse(BaseModel):
    """Overall Sustainability Evaluation output."""
    crop: str = Field(..., description="Evaluated crop.")
    sustainability_index: float = Field(..., description="Composite Sustainability Index (0 to 100).")
    sustainability_grade: str = Field(..., description="'Excellent', 'Good', 'Moderate', or 'Low'.")
    score_breakdown: SustainabilityScoreBreakdown = Field(..., description="Pillar breakdown scores.")
    eco_recommendations: List[str] = Field(..., description="Actionable practices to improve farm sustainability.")
    summary_message: str = Field(..., description="Conversational explanation for the farmer.")


# ============================================================================
# Yield & Profitability Schemas
# ============================================================================

class ProfitabilityRequest(BaseModel):
    """Input parameters to calculate farm yield and economic return."""
    crop: str = Field(..., description="Crop commodity name.", examples=["Rice"])
    land_area_acres: float = Field(
        2.0,
        gt=0.0,
        le=1000.0,
        description="Land size in acres.",
        examples=[2.0]
    )
    mandi_price_per_quintal: Optional[float] = Field(
        None,
        description="Wholesale market price in ₹/quintal. If omitted, fetched automatically.",
        examples=[2350.0]
    )
    budget_inr: Optional[float] = Field(
        None,
        description="Farmer's available working capital in INR.",
        examples=[40000.0]
    )


class ProfitabilityResponse(BaseModel):
    """Yield, financial budget, and profit forecasting details."""
    crop: str = Field(..., description="Crop commodity.")
    land_area_acres: float = Field(..., description="Land area in acres.")
    expected_yield_quintals: float = Field(..., description="Total forecasted harvest yield in quintals.")
    yield_per_acre_quintals: float = Field(..., description="Estimated yield per acre in quintals.")
    estimated_cost_inr: float = Field(..., description="Total cultivation cost in INR.")
    gross_revenue_inr: float = Field(..., description="Total revenue from harvest sale in INR.")
    net_profit_inr: float = Field(..., description="Net financial profit (Revenue - Cost) in INR.")
    roi_percentage: float = Field(..., description="Return on investment percentage.")
    break_even_price_per_quintal: float = Field(..., description="Minimum price per quintal required to avoid loss.")
    summary_message: str = Field(..., description="Farmer-friendly financial summary.")


# ============================================================================
# Crop Rotation Schemas
# ============================================================================

class RotationPlanRequest(BaseModel):
    """Input parameters for seasonal crop rotation planning."""
    current_crop: str = Field(..., description="Current or upcoming primary crop.", examples=["Rice"])
    soil_type: str = Field("Black", description="Soil type (e.g. 'Black', 'Red', 'Loamy').", examples=["Black"])
    irrigation_availability: str = Field(
        "Assured",
        description="'Assured' (Borewell/Canal), 'Moderate', or 'Rainfed'.",
        examples=["Assured"]
    )


class SeasonalRotationStage(BaseModel):
    """Single seasonal crop rotation step."""
    season: str = Field(..., description="'Kharif (Monsoon)', 'Rabi (Winter)', or 'Zaid (Summer)'.")
    recommended_crop: str = Field(..., description="Crop to cultivate in this rotation phase.")
    purpose: str = Field(..., description="Agronomic purpose (e.g. 'Primary Revenue', 'Nitrogen Fixation', 'Rest').")
    duration_days: int = Field(..., description="Approximate crop duration in days.")


class RotationPlanResponse(BaseModel):
    """Complete 1-year sustainable rotation schedule."""
    current_crop: str = Field(..., description="Starting crop.")
    soil_type: str = Field(..., description="Soil type.")
    annual_sequence: List[SeasonalRotationStage] = Field(..., description="Sequential seasonal crop stages.")
    nitrogen_gain_kg_ha: float = Field(..., description="Estimated atmospheric Nitrogen fixed naturally by pulses.")
    pest_cycle_benefit: str = Field(..., description="Pest and weed suppression advantage.")
    summary_message: str = Field(..., description="Conversational rotation explanation.")
