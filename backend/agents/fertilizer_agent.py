"""
Fertilizer Recommendation AI Agent.
Wraps the Fertilizer ML service and agronomic nutrient rules inside a LangChain Tool + Agent.
Interprets farmer queries, identifies nutrient deficiencies, and outputs targeted dosage,
application timing, and sustainable soil health advice.
"""

import os
import re
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

from langchain_core.tools import tool, BaseTool
from backend.schemas.fertilizer_schemas import (
    FertilizerInputSchema,
    FertilizerAgentResponse,
)
from backend.services.fertilizer_service import recommend_fertilizer

load_dotenv()


def create_recommend_fertilizer_tool() -> BaseTool:
    """
    Factory function creating a LangChain Tool for fertilizer recommendation.
    """
    @tool("recommend_fertilizer_tool", args_schema=FertilizerInputSchema)
    def recommend_fertilizer_tool(
        nitrogen: float,
        phosphorus: float,
        potassium: float,
        temperature: float = 26.0,
        humidity: float = 65.0,
        moisture: float = 40.0,
        soil_type: str = "Loamy",
        crop_type: str = "Rice"
    ) -> Dict[str, Any]:
        """
        Recommends specific fertilizer and dosage based on soil NPK levels, soil type, and target crop.
        Requires:
        - nitrogen (kg/ha)
        - phosphorus (kg/ha)
        - potassium (kg/ha)
        Optional: temperature, humidity, moisture, soil_type, crop_type.
        """
        return recommend_fertilizer(
            nitrogen=nitrogen,
            phosphorus=phosphorus,
            potassium=potassium,
            temperature=temperature,
            humidity=humidity,
            moisture=moisture,
            soil_type=soil_type,
            crop_type=crop_type
        )

    return recommend_fertilizer_tool


class FertilizerAgent:
    """
    Conversational agent for soil nutrient evaluation and fertilizer planning.
    """

    MANDATORY_FIELDS: List[str] = ["nitrogen", "phosphorus", "potassium"]

    SOIL_TYPES: List[str] = ["Sandy", "Loamy", "Black", "Red", "Clayey"]
    CROP_TYPES: List[str] = [
        "Rice", "Maize", "Wheat", "Cotton", "Sugarcane",
        "Chickpea", "Kidneybeans", "Pigeonpeas", "Ground Nuts"
    ]

    def __init__(self, llm: Optional[Any] = None):
        """Initialize agent with optional LLM and create LangChain tool."""
        self.tool = create_recommend_fertilizer_tool()
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
                    temperature=0.1
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
                    temperature=0.1
                )
            except Exception:
                pass

        return None

    def extract_parameters_regex(self, text: str) -> Dict[str, Any]:
        """
        Extract N, P, K, soil, crop, moisture, temperature, and humidity using regex.
        Guarantees reliable entity extraction offline.
        """
        lower = text.lower()
        extracted: Dict[str, Any] = {
            "nitrogen": None,
            "phosphorus": None,
            "potassium": None,
            "soil_type": None,
            "crop_type": None,
            "moisture": 40.0,
            "temperature": 26.0,
            "humidity": 65.0
        }

        patterns = {
            "nitrogen": [
                r"(?:nitrogen|\bn\b)\s*(?:is|=|:|\bhas\b|\bof\b)?\s*([0-9]+(?:\.[0-9]+)?)",
                r"([0-9]+(?:\.[0-9]+)?)\s*(?:kg/ha|ppm|%|units?)?\s*(?:of\s+)?(?:nitrogen|\bn\b)",
            ],
            "phosphorus": [
                r"(?:phosphorus|phosphorous|\bp\b)\s*(?:is|=|:|\bhas\b|\bof\b)?\s*([0-9]+(?:\.[0-9]+)?)",
                r"([0-9]+(?:\.[0-9]+)?)\s*(?:kg/ha|ppm|%|units?)?\s*(?:of\s+)?(?:phosphorus|phosphorous|\bp\b)",
            ],
            "potassium": [
                r"(?:potassium|potash|\bk\b)\s*(?:is|=|:|\bhas\b|\bof\b)?\s*([0-9]+(?:\.[0-9]+)?)",
                r"([0-9]+(?:\.[0-9]+)?)\s*(?:kg/ha|ppm|%|units?)?\s*(?:of\s+)?(?:potassium|potash|\bk\b)",
            ],
            "moisture": [
                r"(?:moisture)\s*(?:is|=|:|\bat\b)?\s*([0-9]+(?:\.[0-9]+)?)\s*%?",
                r"([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:moisture)",
            ],
            "temperature": [
                r"(?:temperature|temp)\s*(?:is|=|:|\bat\b)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:°?c|celsius)?",
            ],
            "humidity": [
                r"(?:humidity)\s*(?:is|=|:|\bat\b)?\s*([0-9]+(?:\.[0-9]+)?)\s*%?",
            ]
        }

        for param, regex_list in patterns.items():
            for pat in regex_list:
                match = re.search(pat, lower)
                if match:
                    try:
                        extracted[param] = float(match.group(1))
                        break
                    except (ValueError, IndexError):
                        continue

        # Extract Soil Type
        for soil in self.SOIL_TYPES:
            if re.search(r"\b" + re.escape(soil.lower()) + r"\b", lower):
                extracted["soil_type"] = soil
                break

        # Extract Crop Type
        for crop in self.CROP_TYPES:
            if re.search(r"\b" + re.escape(crop.lower()) + r"\b", lower):
                extracted["crop_type"] = crop
                break
        if not extracted["crop_type"] and re.search(r"\bpaddy\b", lower):
            extracted["crop_type"] = "Rice"

        return extracted

    def identify_missing(self, extracted: Dict[str, Any]) -> List[str]:
        """Detect missing mandatory NPK parameters."""
        return [f for f in self.MANDATORY_FIELDS if extracted.get(f) is None]

    def run(self, query: str) -> FertilizerAgentResponse:
        """
        Process farmer's query and generate fertilizer guidance.
        """
        extracted = self.extract_parameters_regex(query)
        missing = self.identify_missing(extracted)

        if missing:
            missing_names = ", ".join(m.capitalize() for m in missing)
            msg = (
                f"To determine the right fertilizer and dosage, I need your soil's: {missing_names}. "
                "Could you provide your current soil test values for Nitrogen, Phosphorus, and Potassium?"
            )
            return FertilizerAgentResponse(
                status="needs_clarification",
                recommended_fertilizer=None,
                confidence=None,
                dosage_kg_per_acre=None,
                application_schedule=None,
                extracted_parameters=extracted,
                missing_parameters=missing,
                message=msg
            )

        soil = extracted["soil_type"] or "Loamy"
        crop = extracted["crop_type"] or "Rice"

        payload = {
            "nitrogen": float(extracted["nitrogen"]),
            "phosphorus": float(extracted["phosphorus"]),
            "potassium": float(extracted["potassium"]),
            "temperature": float(extracted["temperature"] or 26.0),
            "humidity": float(extracted["humidity"] or 65.0),
            "moisture": float(extracted["moisture"] or 40.0),
            "soil_type": soil,
            "crop_type": crop
        }

        try:
            result = self.tool.invoke(payload)
            fert = result["recommended_fertilizer"]
            conf = float(result["confidence"])
            dose = float(result["dosage_kg_per_acre"])
            schedule = result["application_schedule"]
            report = result["nutrient_report"]
            eco = result["sustainability_tip"]

            advice = (
                f"🌱 Recommended Fertilizer: **{fert}**\n"
                f"• Target Crop: **{crop}** | Soil Type: **{soil}**\n"
                f"• Model Confidence: **{conf*100:.1f}%**\n"
                f"• Limiting Nutrient: **{report['limiting_nutrient']}** (Status: {report[report['limiting_nutrient'].lower().split()[0] + '_status']})\n"
                f"• Recommended Dosage: **{dose} kg per acre**\n"
                f"• Application Schedule: {schedule}\n"
                f"• 🌱 {eco}"
            )

            return FertilizerAgentResponse(
                status="success",
                recommended_fertilizer=fert,
                confidence=conf,
                dosage_kg_per_acre=dose,
                application_schedule=schedule,
                extracted_parameters=extracted,
                missing_parameters=[],
                message=advice
            )
        except Exception as exc:
            return FertilizerAgentResponse(
                status="error",
                recommended_fertilizer=None,
                confidence=None,
                dosage_kg_per_acre=None,
                application_schedule=None,
                extracted_parameters=extracted,
                missing_parameters=[],
                message=f"Error executing fertilizer advisory: {str(exc)}"
            )
