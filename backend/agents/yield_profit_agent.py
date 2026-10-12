"""
Yield & Profitability Evaluation AI Agent (Week 7 / Phase 5).
Forecasts harvest yield, computes production costs, connects live mandi wholesale prices,
and calculates net profit and Return on Investment (ROI %).
"""

import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

from langchain_core.tools import tool, BaseTool
from backend.schemas.evaluation_schemas import (
    ProfitabilityRequest,
    ProfitabilityResponse,
)
from backend.utils.api_clients import MarketPriceClient

load_dotenv()

CROP_FINANCIAL_BENCHMARKS = {
    "rice": {"yield_per_acre": 22.0, "cost_per_acre": 22000.0, "default_price": 2350.0},
    "maize": {"yield_per_acre": 25.0, "cost_per_acre": 18000.0, "default_price": 2150.0},
    "wheat": {"yield_per_acre": 18.0, "cost_per_acre": 16000.0, "default_price": 2420.0},
    "cotton": {"yield_per_acre": 10.0, "cost_per_acre": 28000.0, "default_price": 7120.0},
    "chickpea": {"yield_per_acre": 8.5, "cost_per_acre": 12000.0, "default_price": 5580.0},
    "kidneybeans": {"yield_per_acre": 7.0, "cost_per_acre": 15000.0, "default_price": 8100.0},
    "pigeonpeas": {"yield_per_acre": 7.5, "cost_per_acre": 14000.0, "default_price": 7350.0},
    "ground nuts": {"yield_per_acre": 12.0, "cost_per_acre": 20000.0, "default_price": 6350.0},
    "sugarcane": {"yield_per_acre": 320.0, "cost_per_acre": 45000.0, "default_price": 340.0},
    "default": {"yield_per_acre": 15.0, "cost_per_acre": 16000.0, "default_price": 2500.0}
}


def calculate_profitability_logic(req: ProfitabilityRequest, market_client: Optional[MarketPriceClient] = None) -> ProfitabilityResponse:
    """Core calculation for expected yield, production costs, and net profit."""
    if market_client is None:
        market_client = MarketPriceClient()

    crop_clean = req.crop.strip().lower()
    bench = CROP_FINANCIAL_BENCHMARKS.get(crop_clean, CROP_FINANCIAL_BENCHMARKS["default"])

    acres = req.land_area_acres
    yield_per_acre = bench["yield_per_acre"]
    cost_per_acre = bench["cost_per_acre"]

    total_yield = round(yield_per_acre * acres, 2)
    total_cost = round(cost_per_acre * acres, 2)

    if req.mandi_price_per_quintal and req.mandi_price_per_quintal > 0:
        price_per_quintal = req.mandi_price_per_quintal
    else:
        market_data = market_client.fetch_market_price(req.crop)
        price_per_quintal = float(market_data["modal_price_per_quintal"])

    gross_revenue = round(total_yield * price_per_quintal, 2)
    net_profit = round(gross_revenue - total_cost, 2)
    roi = round((net_profit / total_cost) * 100, 1) if total_cost > 0 else 0.0
    break_even_price = round(total_cost / total_yield, 2) if total_yield > 0 else 0.0

    summary = (
        f"💰 **Financial & Yield Forecast for {req.crop.title()} ({acres} Acres)**\n"
        f"• Estimated Total Yield: **{total_yield} Quintals** ({yield_per_acre} q/acre)\n"
        f"• Market Rate Used: **₹{price_per_quintal:,.2f} per quintal**\n"
        f"• Total Cultivation Cost: **₹{total_cost:,.2f}**\n"
        f"• Estimated Gross Revenue: **₹{gross_revenue:,.2f}**\n"
        f"• Expected Net Profit: **₹{net_profit:,.2f}**\n"
        f"• Return on Investment (ROI): **{roi}%** (Break-even rate: ₹{break_even_price}/q)"
    )

    return ProfitabilityResponse(
        crop=req.crop.title(),
        land_area_acres=acres,
        expected_yield_quintals=total_yield,
        yield_per_acre_quintals=yield_per_acre,
        estimated_cost_inr=total_cost,
        gross_revenue_inr=gross_revenue,
        net_profit_inr=net_profit,
        roi_percentage=roi,
        break_even_price_per_quintal=break_even_price,
        summary_message=summary
    )


def create_profitability_tool() -> BaseTool:
    """Factory function creating a LangChain tool for yield and profit estimation."""
    @tool("calculate_profitability_tool", args_schema=ProfitabilityRequest)
    def calculate_profitability_tool(
        crop: str,
        land_area_acres: float = 2.0,
        mandi_price_per_quintal: Optional[float] = None,
        budget_inr: Optional[float] = None
    ) -> Dict[str, Any]:
        """Calculates forecasted crop yield, cultivation expenditure, gross revenue, net profit, and ROI."""
        req = ProfitabilityRequest(
            crop=crop,
            land_area_acres=land_area_acres,
            mandi_price_per_quintal=mandi_price_per_quintal,
            budget_inr=budget_inr
        )
        return calculate_profitability_logic(req).model_dump()

    return calculate_profitability_tool


class YieldProfitAgent:
    """Agent specializing in agronomic yield forecasting and farm financial return analysis."""

    def __init__(self, llm: Optional[Any] = None):
        """Initialize agent with market pricing client."""
        self.market_client = MarketPriceClient()
        self.tool = create_profitability_tool()
        self.llm = llm

    def evaluate(self, req: ProfitabilityRequest) -> ProfitabilityResponse:
        """Compute estimated yield, gross revenue, cultivation expenditure, and net profit."""
        return calculate_profitability_logic(req, self.market_client)

    def run(self, query: str) -> ProfitabilityResponse:
        """Process natural text query regarding crop profit or revenue."""
        crop_clean = "Rice"
        for candidate in ["Cotton", "Maize", "Wheat", "Chickpea", "Sugarcane", "Rice"]:
            if candidate.lower() in query.lower():
                crop_clean = candidate
                break

        acres = 2.0
        import re
        match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(?:acres?|acre)", query.lower())
        if match:
            acres = float(match.group(1))

        req = ProfitabilityRequest(crop=crop_clean, land_area_acres=acres)
        return self.evaluate(req)
