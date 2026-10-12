"""
Pydantic schemas for Multi-Agent Orchestrator and Reasoning Engine (Week 8 / Phase 6).
Defines contracts for comprehensive end-to-end agronomic queries and multi-agent synthesis packages.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class OrchestratorQueryRequest(BaseModel):
    """Farmer's natural language input along with optional farm metadata."""
    query: str = Field(
        ...,
        description="Farmer's complete natural language query.",
        examples=["My soil has 90 nitrogen, 42 phosphorus, 43 potassium, temperature is 25°C, humidity 80%, pH 6.5, rainfall 200mm. What should I grow?"]
    )
    location: Optional[str] = Field(
        None,
        description="Optional city/district name if not mentioned in text.",
        examples=["Nizamabad, Telangana"]
    )
    land_area_acres: Optional[float] = Field(
        2.0,
        ge=0.1,
        le=1000.0,
        description="Farm size in acres.",
        examples=[2.0]
    )
    irrigation_method: Optional[str] = Field(
        "Flood",
        description="Irrigation method: 'Drip', 'Sprinkler', 'Flood', 'Rainfed'.",
        examples=["Drip"]
    )


class OrchestratorResponse(BaseModel):
    """Complete multi-agent advisory package synthesized for the farmer."""
    status: str = Field(..., description="'success', 'clarification_needed', or 'error'.")
    primary_recommendation: Dict[str, Any] = Field(..., description="Crop and fertilizer predictions.")
    environmental_context: Dict[str, Any] = Field(..., description="Live weather and regional soil data.")
    market_and_profitability: Dict[str, Any] = Field(..., description="Mandi wholesale prices, expected yield, and net profit.")
    sustainability_and_planning: Dict[str, Any] = Field(..., description="Sustainability score and 1-year crop rotation sequence.")
    explainable_reasoning: str = Field(..., description="Phase 6 farmer-friendly synthesized reasoning.")
    missing_information: List[str] = Field(default_factory=list, description="Any parameters missing from query.")
