"""
Soil AI Agent.
Autonomous agent that retrieves regional soil nutrient baselines, dominant soil types,
and pH estimates when a farmer lacks a recent soil health card.
"""

import os
import re
from typing import Dict, Any, Optional
from dotenv import load_dotenv

from langchain_core.tools import tool, BaseTool
from backend.schemas.data_schemas import (
    SoilAgentRequest,
    SoilProfileResponse,
)
from backend.utils.api_clients import SoilClient

load_dotenv()


def create_soil_tool() -> BaseTool:
    """Factory function creating a LangChain Tool for soil profile retrieval."""
    client = SoilClient()

    @tool("get_soil_profile_tool", args_schema=SoilAgentRequest)
    def get_soil_profile_tool(location: str) -> Dict[str, Any]:
        """
        Retrieves dominant soil classification, pH, organic carbon, and baseline NPK
        for a given state, district, or agro-climatic zone.
        """
        return client.fetch_soil_profile(location)

    return get_soil_profile_tool


class SoilAgent:
    """Independent agent providing regional soil characteristics and management guidance."""

    KNOWN_REGIONS = [
        "nizamabad", "hyderabad", "telangana", "andhra", "guntur",
        "punjab", "ludhiana", "maharashtra", "pune", "karnataka"
    ]

    def __init__(self, llm: Optional[Any] = None):
        """Initialize soil agent with client and tool."""
        self.tool = create_soil_tool()
        self.client = SoilClient()
        self.llm = llm

    def extract_location(self, query: str) -> str:
        """Extract regional district or state name from text."""
        clean = query.strip()
        lower = clean.lower()

        for reg in self.KNOWN_REGIONS:
            if re.search(r"\b" + re.escape(reg) + r"\b", lower):
                return reg.title()

        match = re.search(r"\b(?:in|at|around|for|near)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)", clean)
        if match:
            candidate = match.group(1).strip()
            if candidate.lower() not in ["soil", "my", "land", "field", "farm"]:
                return candidate.title()

        return "Telangana"

    def run(self, query: str) -> SoilProfileResponse:
        """
        Process farmer's soil query and return regional soil profile.
        """
        location = self.extract_location(query)
        raw = self.tool.invoke({"location": location})

        reg = str(raw["region"])
        stype = str(raw["dominant_soil_type"])
        ph = float(raw["average_ph"])
        n = float(raw["average_nitrogen"])
        p = float(raw["average_phosphorus"])
        k = float(raw["average_potassium"])
        oc = float(raw["organic_carbon_percent"])
        source = str(raw["source"])

        ph_eval = "Slightly Acidic" if ph < 6.5 else ("Alkaline" if ph > 7.5 else "Near Neutral (Optimal)")

        summary = (
            f"🌱 **Regional Soil Profile for {reg}**\n"
            f"• Dominant Soil: **{stype} Soil**\n"
            f"• Estimated pH: **{ph}** ({ph_eval})\n"
            f"• Baseline N-P-K: **{n:.0f} N - {p:.0f} P - {k:.0f} K (kg/ha)**\n"
            f"• Organic Carbon: **{oc:.2f}%**\n"
            f"• Database Source: {source}\n"
            f"• Recommendation: Suitable for crops adapted to {stype.lower()} soils with high cation exchange."
        )

        return SoilProfileResponse(
            region=reg,
            dominant_soil_type=stype,
            average_ph=ph,
            average_nitrogen=n,
            average_phosphorus=p,
            average_potassium=k,
            organic_carbon_percent=oc,
            source=source,
            message=summary
        )
