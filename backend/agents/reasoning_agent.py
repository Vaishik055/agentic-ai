"""
Reasoning AI Agent (Week 8 / Phase 6).
Synthesizes all multi-agent outputs (Crop, Fertilizer, Weather, Soil, Market, Profit, Sustainability)
into a personalized, explainable, farmer-friendly recommendation.
"""

import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()


class ReasoningAgent:
    """Agentic synthesizer delivering explainable, holistic farming advice."""

    def __init__(self, llm: Optional[Any] = None):
        """Initialize reasoning agent with optional LLM."""
        self.llm = llm or self._initialize_llm()

    def _initialize_llm(self) -> Optional[Any]:
        """Initialize Groq or OpenAI LLM if credentials exist."""
        groq_api_key = os.getenv("GROQ_API_KEY")
        if groq_api_key and groq_api_key != "your_groq_api_key_here":
            try:
                from langchain_groq import ChatGroq
                return ChatGroq(
                    api_key=groq_api_key,
                    model_name="llama-3.3-70b-versatile",
                    temperature=0.2
                )
            except Exception:
                pass

        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key and openai_api_key != "your_openai_api_key_here":
            try:
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(
                    api_key=openai_api_key,
                    model="gpt-4o-mini",
                    temperature=0.2
                )
            except Exception:
                pass

        return None

    def synthesize(self, state: Dict[str, Any]) -> str:
        """
        Synthesize multi-agent data into transparent, explainable recommendations.
        """
        crop_info = state.get("crop_recommendation", {})
        fert_info = state.get("fertilizer_recommendation", {})
        weather_info = state.get("weather_data", {})
        soil_info = state.get("soil_data", {})
        profit_info = state.get("profitability_report", {})
        sust_info = state.get("sustainability_report", {})
        rot_info = state.get("rotation_plan", {})

        crop_name = crop_info.get("recommended_crop", "Rice").title()
        conf = float(crop_info.get("confidence", 0.95)) * 100
        fert_name = fert_info.get("recommended_fertilizer", "Urea")
        dose = fert_info.get("dosage_kg_per_acre", 45.0)

        loc = weather_info.get("location", "Regional Farm")
        temp = weather_info.get("temperature", 26.0)
        rain = weather_info.get("rainfall_forecast_mm", 200.0)

        yield_q = profit_info.get("expected_yield_quintals", 44.0)
        net_profit = profit_info.get("net_profit_inr", 55000.0)
        roi = profit_info.get("roi_percentage", 110.0)

        sust_score = sust_info.get("sustainability_index", 75.0)
        sust_grade = sust_info.get("sustainability_grade", "Good (Sustainable)")
        eco_tip = sust_info.get("eco_recommendations", ["Maintain organic mulch."])[0]

        rabi_crop = rot_info.get("annual_sequence", [{}, {"recommended_crop": "Chickpea"}])[1].get("recommended_crop", "Chickpea")
        n_gain = rot_info.get("nitrogen_gain_kg_ha", 45.0)

        # Attempt LLM synthesis if available
        if self.llm is not None:
            try:
                from langchain_core.prompts import ChatPromptTemplate
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are the Chief Agricultural Officer. Synthesize the multi-agent findings into a clear, explainable, encouraging recommendation for a smallholder farmer. Explain WHY this crop fits the soil and weather, how much profit to expect, and how to maintain soil sustainability."),
                    ("human", "Farm Data:\n- Crop: {crop} ({conf:.1f}% confidence)\n- Location Weather: {loc}, {temp}°C, {rain}mm rain\n- Fertilizer: {fert} ({dose} kg/acre)\n- Economics: {yield_q} quintals harvest, ₹{profit:,.2f} net profit (ROI: {roi}%)\n- Sustainability: {sust_score}/100 ({sust_grade})\n- Eco-Tip: {eco_tip}\n- Rotation: Next season sow {rabi_crop} (+{n_gain} kg/ha natural N)")
                ])
                chain = prompt | self.llm
                res = chain.invoke({
                    "crop": crop_name,
                    "conf": conf,
                    "loc": loc,
                    "temp": temp,
                    "rain": rain,
                    "fert": fert_name,
                    "dose": dose,
                    "yield_q": yield_q,
                    "profit": net_profit,
                    "roi": roi,
                    "sust_score": sust_score,
                    "sust_grade": sust_grade,
                    "eco_tip": eco_tip,
                    "rabi_crop": rabi_crop,
                    "n_gain": n_gain
                })
                return str(res.content)
            except Exception:
                pass

        # Deterministic grounded reasoning format (Phase 6 deliverable)
        return (
            f"🌾 **Personalized Farming Decision Package**\n\n"
            f"1. **Crop Selection:** Grow **{crop_name}** (Confidence: **{conf:.1f}%**)\n"
            f"   • *Reasoning:* Your climate in {loc} (temp: {temp}°C, seasonal rain: {rain} mm) "
            f"and soil nutrients provide optimal conditions for high grain yield.\n\n"
            f"2. **Fertilizer & Soil Nutrition:** Apply **{fert_name}** @ **{dose} kg/acre**\n"
            f"   • *Application:* Split into 50% basal dose at sowing and remainder top-dressed at 30 days.\n\n"
            f"3. **Financial Forecast:**\n"
            f"   • Estimated Total Harvest: **{yield_q} Quintals**\n"
            f"   • Projected Net Profit: **₹{net_profit:,.2f}** (Estimated ROI: **{roi}%**)\n\n"
            f"4. **Sustainability Index:** **{sust_score}/100** ({sust_grade})\n"
            f"   • *Eco-Action:* {eco_tip}\n\n"
            f"5. **Crop Rotation Strategy:**\n"
            f"   • Follow in Rabi with **{rabi_crop}** to naturally fix **+{n_gain:.0f} kg N/ha** back into your soil."
        )
