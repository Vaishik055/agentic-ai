"""
Sustainability Evaluation AI Agent (Week 7 / Phase 5).
Computes a comprehensive 0-100 Sustainability Index across 4 ecological pillars:
Water footprint, Soil health preservation, Chemical load, and Carbon/Climate impact.
"""

import os
from typing import Dict, Any, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from langchain_core.tools import tool, BaseTool
except ImportError:
    def tool(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator
    class BaseTool:
        pass

from backend.schemas.evaluation_schemas import (
    SustainabilityRequest,
    SustainabilityResponse,
    SustainabilityScoreBreakdown,
)


def evaluate_sustainability_logic(req: SustainabilityRequest) -> SustainabilityResponse:
    """Core deterministic calculation for farm sustainability metrics."""
    crop_clean = req.crop.strip().title()
    irr_clean = req.irrigation_method.strip().title()
    fert_clean = req.fertilizer_used.strip().title()
    soc = req.soil_organic_carbon
    rain = req.rainfall_mm

    recommendations: List[str] = []

    # 1. Water Efficiency Score (Max 25 pts)
    if irr_clean == "Drip":
        water_score = 25.0
    elif irr_clean == "Sprinkler":
        water_score = 21.0
    elif irr_clean == "Rainfed":
        water_score = 22.0 if rain >= 120.0 else 16.0
    else:  # Flood irrigation
        water_score = 12.0
        recommendations.append(
            "Adopt Alternate Wetting & Drying (AWD) or Drip Irrigation to cut water usage by 25-35%."
        )

    # 2. Soil Health Score (Max 25 pts)
    if soc >= 0.75:
        soil_score = 25.0
    elif soc >= 0.50:
        soil_score = 19.0
    else:
        soil_score = 12.0
        recommendations.append(
            "Soil organic carbon is below 0.50%. Apply 2 tons/acre of compost or vermicompost to restore soil fertility."
        )

    # 3. Chemical Load Score (Max 25 pts)
    if any(term in fert_clean.lower() for term in ["organic", "bio", "manure", "vermicompost"]):
        chem_score = 25.0
    elif "neem" in fert_clean.lower():
        chem_score = 22.0
    elif fert_clean in ["10-26-26", "20-20-0", "14-35-14"]:
        chem_score = 18.0
    elif fert_clean in ["DAP", "Urea"]:
        chem_score = 14.0
        recommendations.append(
            "High synthetic NPK load detected. Integrate bio-fertilizers (Azospirillum / PSB) to lower chemical runoff."
        )
    else:
        chem_score = 16.0

    # 4. Carbon & Practice Score (Max 25 pts)
    carbon_score = 15.0
    if req.residue_burning:
        carbon_score -= 8.0
        recommendations.append(
            "Strictly avoid crop residue burning; incorporate stubble back into soil or use Happy Seeder."
        )
    else:
        carbon_score += 4.0

    if req.organic_amendments:
        carbon_score += 4.0
    if req.cover_cropping:
        carbon_score += 2.0

    carbon_score = max(5.0, min(25.0, carbon_score))

    # Total Index
    total_score = round(water_score + soil_score + chem_score + carbon_score, 1)

    # Grade Assignment
    if total_score >= 80.0:
        grade = "A (Highly Sustainable)"
    elif total_score >= 65.0:
        grade = "B (Sustainable)"
    elif total_score >= 50.0:
        grade = "C (Moderate / Needs Attention)"
    else:
        grade = "D (Unsustainable / High Risk)"

    if not recommendations:
        recommendations.append("Excellent agronomic practices! Maintain current organic and moisture conservation protocols.")

    return SustainabilityResponse(
        sustainability_index=total_score,
        grade=grade,
        breakdown=SustainabilityScoreBreakdown(
            water_efficiency_score=water_score,
            soil_health_score=soil_score,
            chemical_load_score=chem_score,
            carbon_practice_score=carbon_score,
        ),
        recommendations=recommendations,
    )


def create_sustainability_tool() -> BaseTool:
    """Factory function creating LangChain tool for sustainability scoring."""

    @tool("evaluate_sustainability_tool", args_schema=SustainabilityRequest)
    def evaluate_sustainability_tool(
        crop: str,
        irrigation_method: str = "Flood",
        fertilizer_used: str = "Urea",
        soil_organic_carbon: float = 0.55,
        rainfall_mm: float = 100.0,
        residue_burning: bool = False,
        organic_amendments: bool = False,
        cover_cropping: bool = False,
    ) -> Dict[str, Any]:
        """Evaluates farm sustainability index (0-100) and returns eco-friendly recommendations."""
        req = SustainabilityRequest(
            crop=crop,
            irrigation_method=irrigation_method,
            fertilizer_used=fertilizer_used,
            soil_organic_carbon=soil_organic_carbon,
            rainfall_mm=rainfall_mm,
            residue_burning=residue_burning,
            organic_amendments=organic_amendments,
            cover_cropping=cover_cropping,
        )
        return evaluate_sustainability_logic(req).model_dump()

    return evaluate_sustainability_tool


class SustainabilityAgent:
    """Agent evaluating agronomic decisions against environmental and sustainability benchmarks."""

    def __init__(self):
        self.tool = create_sustainability_tool()

    def evaluate(self, request: SustainabilityRequest) -> SustainabilityResponse:
        """Direct computation of sustainability score."""
        return evaluate_sustainability_logic(request)

    def run(self, query: str) -> Dict[str, Any]:
        """Conversational wrapper for agent evaluation."""
        req = SustainabilityRequest(
            crop="Rice",
            irrigation_method="Flood",
            fertilizer_used="Urea",
            soil_organic_carbon=0.55,
            rainfall_mm=100.0,
            residue_burning=False,
            organic_amendments=False,
            cover_cropping=False,
        )
        return evaluate_sustainability_logic(req).model_dump()