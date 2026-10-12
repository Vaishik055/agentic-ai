"""
Weather AI Agent.
Autonomous agent that extracts location names from farmer queries,
fetches live/regional meteorological parameters (temperature, humidity, rainfall),
and formats agronomic weather intelligence.
"""

import os
import re
from typing import Dict, Any, Optional
from dotenv import load_dotenv

from langchain_core.tools import tool, BaseTool
from backend.schemas.data_schemas import (
    WeatherAgentRequest,
    WeatherDataResponse,
)
from backend.utils.api_clients import WeatherClient

load_dotenv()


def create_weather_tool() -> BaseTool:
    """Factory function creating a LangChain Tool for weather retrieval."""
    client = WeatherClient()

    @tool("get_weather_tool", args_schema=WeatherAgentRequest)
    def get_weather_tool(location: str) -> Dict[str, Any]:
        """
        Retrieves real-time or agro-climatic temperature, humidity, and rainfall forecast
        for any district, city, or agricultural region.
        """
        return client.fetch_weather(location)

    return get_weather_tool


class WeatherAgent:
    """Independent agent specialized in agro-meteorological retrieval and advice."""

    KNOWN_LOCATIONS = [
        "hyderabad", "nizamabad", "guntur", "pune", "ludhiana",
        "varanasi", "bengaluru", "delhi", "mumbai", "chennai", "kolkata"
    ]

    def __init__(self, llm: Optional[Any] = None):
        """Initialize weather agent with client and LangChain tool."""
        self.tool = create_weather_tool()
        self.client = WeatherClient()
        self.llm = llm

    def extract_location(self, query: str) -> str:
        """Extract location name from farmer text query."""
        clean = query.strip()
        lower = clean.lower()

        # Check for known agricultural hubs
        for loc in self.KNOWN_LOCATIONS:
            if re.search(r"\b" + re.escape(loc) + r"\b", lower):
                return loc.title()

        # Regex patterns for location phrases like "in Nizamabad", "at Guntur", "for Hyderabad"
        match = re.search(r"\b(?:in|at|near|around|for)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)", clean)
        if match:
            candidate = match.group(1).strip()
            # Exclude non-location common words
            if candidate.lower() not in ["soil", "rice", "wheat", "crop", "farming", "my", "this"]:
                return candidate.title()

        return "Hyderabad"  # Sensible default regional hub

    def run(self, query: str) -> WeatherDataResponse:
        """
        Process farmer's weather query and return structured meteorological data.
        """
        location = self.extract_location(query)
        raw_data = self.tool.invoke({"location": location})

        temp = float(raw_data["temperature"])
        humidity = float(raw_data["humidity"])
        rain = float(raw_data["rainfall_forecast_mm"])
        condition = str(raw_data["weather_condition"])
        loc_name = str(raw_data["location"])
        source = str(raw_data["source"])

        # Determine agronomic impact
        if rain >= 180 and humidity >= 70:
            agri_note = "High moisture and ample rainfall — optimal for Kharif paddy, jute, and sugarcane."
        elif rain < 100:
            agri_note = "Lower rainfall zone — consider drip irrigation or drought-resilient crops (millets, pulses)."
        else:
            agri_note = "Moderate climate — favorable for maize, cotton, and groundnut cultivation."

        summary = (
            f"🌤️ **Weather Forecast for {loc_name}**\n"
            f"• Temperature: **{temp}°C** | Humidity: **{humidity}%**\n"
            f"• Rainfall Forecast: **{rain} mm** | Condition: **{condition}**\n"
            f"• Data Source: {source}\n"
            f"• Agronomic Impact: {agri_note}"
        )

        return WeatherDataResponse(
            location=loc_name,
            temperature=temp,
            humidity=humidity,
            rainfall_forecast_mm=rain,
            weather_condition=condition,
            source=source,
            message=summary
        )
