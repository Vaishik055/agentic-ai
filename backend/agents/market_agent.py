"""
Market Price AI Agent.
Autonomous agent that retrieves live wholesale mandi commodity rates (₹/quintal)
from Agmarknet / APMC markets to support profitability evaluation.
"""

import os
import re
from typing import Dict, Any, Optional
from dotenv import load_dotenv

from langchain_core.tools import tool, BaseTool
from backend.schemas.data_schemas import (
    MarketAgentRequest,
    MarketPriceResponse,
)
from backend.utils.api_clients import MarketPriceClient

load_dotenv()


def create_market_tool() -> BaseTool:
    """Factory function creating a LangChain Tool for mandi commodity pricing."""
    client = MarketPriceClient()

    @tool("get_mandi_prices_tool", args_schema=MarketAgentRequest)
    def get_mandi_prices_tool(crop: str, state: Optional[str] = "Telangana") -> Dict[str, Any]:
        """
        Retrieves current mandi wholesale commodity prices (Min, Modal, Max ₹/quintal)
        for any crop and Indian state.
        """
        return client.fetch_market_price(crop=crop, state=state or "Telangana")

    return get_mandi_prices_tool


class MarketAgent:
    """Independent agent specialized in agricultural market pricing and crop economics."""

    COMMODITIES = [
        "rice", "paddy", "wheat", "maize", "cotton", "chickpea",
        "kidneybeans", "pigeonpeas", "ground nuts", "sugarcane"
    ]

    STATES = ["telangana", "andhra", "punjab", "maharashtra", "karnataka", "uttar pradesh"]

    def __init__(self, llm: Optional[Any] = None):
        """Initialize market agent with client and tool."""
        self.tool = create_market_tool()
        self.client = MarketPriceClient()
        self.llm = llm

    def extract_crop_and_state(self, query: str) -> tuple[str, str]:
        """Extract crop name and optional state from text."""
        lower = query.lower()

        matched_crop = "Rice"
        for c in self.COMMODITIES:
            if re.search(r"\b" + re.escape(c) + r"\b", lower):
                matched_crop = c.title()
                break

        matched_state = "Telangana"
        for s in self.STATES:
            if re.search(r"\b" + re.escape(s) + r"\b", lower):
                matched_state = s.title()
                break

        return matched_crop, matched_state

    def run(self, query: str) -> MarketPriceResponse:
        """
        Process farmer's market pricing query and return prevailing wholesale rates.
        """
        crop, state = self.extract_crop_and_state(query)
        raw = self.tool.invoke({"crop": crop, "state": state})

        commodity = str(raw["commodity"])
        st = str(raw["state"])
        mandi = str(raw["market_mandi"])
        p_min = float(raw["min_price_per_quintal"])
        p_modal = float(raw["modal_price_per_quintal"])
        p_max = float(raw["max_price_per_quintal"])
        unit = str(raw["unit"])
        source = str(raw["source"])

        summary = (
            f"📈 **Mandi Wholesale Rates for {commodity} in {st}**\n"
            f"• Reporting Market: **{mandi}**\n"
            f"• Prevailing Modal Price: **₹{p_modal:,.2f} per quintal** (₹{p_modal/100:.2f}/kg)\n"
            f"• Trading Price Range: ₹{p_min:,.2f} – ₹{p_max:,.2f} / quintal\n"
            f"• Price Unit: {unit}\n"
            f"• Source: {source}\n"
            f"• Economic Outlook: Favorable wholesale demand across regional primary APMCs."
        )

        return MarketPriceResponse(
            commodity=commodity,
            state=st,
            market_mandi=mandi,
            min_price_per_quintal=p_min,
            modal_price_per_quintal=p_modal,
            max_price_per_quintal=p_max,
            unit=unit,
            source=source,
            message=summary
        )
