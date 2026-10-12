"""
Sustainability Evaluation AI Agent (Week 7 / Phase 5).
Computes a comprehensive 0-100 Sustainability Index across 4 ecological pillars:
Water footprint, Soil health preservation, Chemical load, and Carbon/Climate impact.
"""

import os
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

from langchain_core.tools import tool, BaseTool
from backend.schemas.evaluation_schemas import (
    SustainabilityRequest,
    SustainabilityResponse,
    SustainabilityScoreBreakdown,
)

load_dotenv()


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
        chem_score = 21.0
    elif any(f in fert_clean for f in ["10-26-26", "17-17-17", "20-20"]):
        chem_score = 18.0
    else:
        chem_score = 14.0
        recommendations.append(
            "Substitute plain urea with Neem-Coated Urea and integrate Phosphate Solubilizing Bacteria (PSB)."
        )

    # 4. Carbon & Climate Score (Max 25 pts)
    is_legume = any(leg in crop_clean.lower() for leg in ["chickpea", "pigeonpea", "gram", "bean", "lentil", "mung"])
    if is_legume:
        carbon_score = 25.0
    elif crop_clean.lower() in ["maize", "wheat", "cotton"]:
        carbon_score = 20.0
    elif crop_clean.lower() == "rice" and irr_clean == "Flood":
        carbon_score = 14.0
        recommendations.append(
            "Practice periodic aeration during paddy cultivation to suppress anaerobic methane emissions."
        )
    else:
        carbon_score = 18.0

    total_score = round(water_score + soil_score + chem_score + carbon_score, 1)

    if total_score >= 80.0:
        grade = "Excellent (Eco-Leader)"
    elif total_score >= 65.0:
        grade = "Good (Sustainable)"
    elif total_score >= 50.0:
        grade = "Moderate (Improvement Needed)"
    else:
        grade = "Low (Ecological Risk)"

    if not recommendations:
        recommendations.append("Maintain current eco-friendly practices and continue green manure crop rotation.")

    summary = (
        f"🌿 **Farm Sustainability Evaluation for {crop_clean}**\n"
        f"• Composite Sustainability Index: **{total_score}/100** ({grade})\n"
        f"• Water Efficiency: **{water_score}/25** | Soil Health: **{soil_score}/25**\n"
        f"• Chemical Load: **{chem_score}/25** | Carbon Footprint: **{carbon_score}/25**\n"
        f"• Key Recommendation: {recommendations[0]}"
    )

    return SustainabilityResponse(
        crop=crop_clean,
        sustainability_index=total_score,
        sustainability_grade=grade,
        score_breakdown=SustainabilityScoreBreakdown(
            water_efficiency_score=water_score,
            soil_health_score=soil_score,
            chemical_load_score=chem_score,
            carbon_climate_score=carbon_score
        ),
        eco_recommendations=recommendations,
        summary_message=summary
    )


def create_sustainability_tool() -> BaseTool:
    """Factory function creating a LangChain tool for sustainability scoring."""
    @tool("evaluate_sustainability_tool", args_schema=SustainabilityRequest)
    def evaluate_sustainability_tool(
        crop: str,
        irrigation_method: str = "Flood",
        fertilizer_used: str = "Urea",
        soil_organic_carbon: float = 0.55,
        rainfall_mm: float = 200.0
    ) -> Dict[str, Any]:
        """Calculates a 0-100 Sustainability Index and recommends eco-friendly adjustments."""
        req = SustainabilityRequest(
            crop=crop,
            irrigation_method=irrigation_method,
            fertilizer_used=fertilizer_used,
            soil_organic_carbon=soil_organic_carbon,
            rainfall_mm=rainfall_mm
        )
        return evaluate_sustainability_logic(req).model_dump()

    return evaluate_sustainability_tool


class SustainabilityAgent:
    """Independent agent assessing environmental impacts and ecological sustainability."""

    def __init__(self, llm: Optional[Any] = None):
        """Initialize agent with optional LLM."""
        self.tool = create_sustainability_tool()
        self.llm = llm

    def evaluate(self, req: SustainabilityRequest) -> SustainabilityResponse:
        """Evaluate sustainability using core agronomic rules."""
        return evaluate_sustainability_logic(req)

    def run(self, query_or_crop: str) -> SustainabilityResponse:
        """Process farmer's sustainability evaluation query."""
        crop_clean = "Rice"
        for candidate in ["Cotton", "Maize", "Wheat", "Chickpea", "Sugarcane", "Rice"]:
            if candidate.lower() in query_or_crop.lower():
                crop_clean = candidate
                break

        irr = "Drip" if "drip" in query_or_crop.lower() else "Flood"
        fert = "Neem-Coated Urea" if "neem" in query_or_crop.lower() else "Urea"

        req = SustainabilityRequest(
            crop=crop_clean,
            irrigation_method=irr,
            fertilizer_used=fert,
            soil_organic_carbon=0.55,
            rainfall_mm=200.0
        )
        return self.evaluate(req)
