"""
Crop Rotation & Seasonal Farm Planner AI Agent (Week 7 / Phase 4 & 5).
Designs multi-season sustainable rotation sequences to break pest cycles,
minimize chemical inputs, and fix natural atmospheric nitrogen into the soil.
"""

from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

from langchain_core.tools import tool, BaseTool
from backend.schemas.evaluation_schemas import (
    RotationPlanRequest,
    RotationPlanResponse,
    SeasonalRotationStage,
)

load_dotenv()

ROTATION_TEMPLATES = {
    "rice": {
        "rabi": ("Chickpea (Gram)", "Nitrogen Fixation & Moisture Conservation", 110, 45.0),
        "zaid": ("Green Gram (Mungbean) / Green Manure", "Soil Restoration & Microclimate Cover", 65, 35.0)
    },
    "cotton": {
        "rabi": ("Wheat or Chickpea", "Breaks Pink Bollworm Lifecycle & Grain Production", 120, 40.0),
        "zaid": ("Sesame or Fallow / Green Manuring", "Soil Solarization & Weed Management", 70, 25.0)
    },
    "maize": {
        "rabi": ("Mustard or Chickpea", "Low Water Requirement & Crop Diversification", 100, 35.0),
        "zaid": ("Cowpea (Lobia)", "Organic Biomass & Nitrogen Enrichment", 60, 30.0)
    },
    "default": {
        "rabi": ("Chickpea / Pulses", "Root Nodulation & Nitrogen Fixation", 105, 40.0),
        "zaid": ("Dhaincha (Sesbania)", "Green Manure Incorporation", 45, 50.0)
    }
}


def plan_rotation_logic(req: RotationPlanRequest) -> RotationPlanResponse:
    """Core logic to construct sequential rotation stages and compute soil benefits."""
    crop_clean = req.current_crop.strip().title()
    template = ROTATION_TEMPLATES.get(crop_clean.lower(), ROTATION_TEMPLATES["default"])

    stages = [
        SeasonalRotationStage(
            season="Kharif (Monsoon: June - Oct)",
            recommended_crop=crop_clean,
            purpose="Primary Commercial Harvest & Revenue Generation",
            duration_days=130
        ),
        SeasonalRotationStage(
            season="Rabi (Winter: Nov - March)",
            recommended_crop=template["rabi"][0],
            purpose=template["rabi"][1],
            duration_days=template["rabi"][2]
        ),
        SeasonalRotationStage(
            season="Zaid (Summer: April - June)",
            recommended_crop=template["zaid"][0],
            purpose=template["zaid"][1],
            duration_days=template["zaid"][2]
        )
    ]

    n_gain = template["rabi"][3] + template["zaid"][3]
    pest_benefit = (
        f"Rotating {crop_clean} with {template['rabi'][0]} breaks dominant monophagous pest cycles "
        "(such as stem borer and bollworm) and suppresses weed seed banks by 40%."
    )

    summary = (
        f"🔄 **Sustainable 1-Year Crop Rotation Plan for {crop_clean}**\n"
        f"1. **Kharif:** {crop_clean} (Main Commercial Crop)\n"
        f"2. **Rabi:** {template['rabi'][0]} ({template['rabi'][1]})\n"
        f"3. **Zaid:** {template['zaid'][0]} ({template['zaid'][1]})\n"
        f"• Natural Nitrogen Enrichment: **+{n_gain:.0f} kg N/ha** fixed biologically\n"
        f"• Agro-Ecological Impact: {pest_benefit}"
    )

    return RotationPlanResponse(
        current_crop=crop_clean,
        soil_type=req.soil_type,
        annual_sequence=stages,
        nitrogen_gain_kg_ha=n_gain,
        pest_cycle_benefit=pest_benefit,
        summary_message=summary
    )


def create_rotation_tool() -> BaseTool:
    """Factory function creating a LangChain tool for crop rotation planning."""
    @tool("plan_crop_rotation_tool", args_schema=RotationPlanRequest)
    def plan_crop_rotation_tool(
        current_crop: str,
        soil_type: str = "Black",
        irrigation_availability: str = "Assured"
    ) -> Dict[str, Any]:
        """Generates an agronomic 1-year multi-season crop rotation plan."""
        req = RotationPlanRequest(
            current_crop=current_crop,
            soil_type=soil_type,
            irrigation_availability=irrigation_availability
        )
        return plan_rotation_logic(req).model_dump()

    return plan_crop_rotation_tool


class RotationPlannerAgent:
    """Agent designing multi-season agronomic crop rotation sequences."""

    def __init__(self, llm: Optional[Any] = None):
        """Initialize rotation planner agent."""
        self.tool = create_rotation_tool()
        self.llm = llm

    def plan_rotation(self, req: RotationPlanRequest) -> RotationPlanResponse:
        """Construct sequential rotation stages and compute soil benefits."""
        return plan_rotation_logic(req)

    def run(self, query: str) -> RotationPlanResponse:
        """Process natural text query for crop rotation advice."""
        crop_clean = "Rice"
        for candidate in ["Cotton", "Maize", "Wheat", "Chickpea", "Rice"]:
            if candidate.lower() in query.lower():
                crop_clean = candidate
                break

        req = RotationPlanRequest(current_crop=crop_clean, soil_type="Black", irrigation_availability="Assured")
        return self.plan_rotation(req)
