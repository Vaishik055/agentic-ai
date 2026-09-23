"""
Crop Recommendation AI Agent.
Wraps the Week 2 Random Forest ML service inside a LangChain Tool + Agent.
Processes unstructured natural language queries from farmers, extracts soil/weather
parameters, detects missing data to ask clarifying questions, and delivers
explainable crop recommendations.
"""

import os
import re
from typing import Dict, Any, List, Optional, Tuple
from dotenv import load_dotenv

from langchain_core.tools import tool, BaseTool
from backend.schemas.crop_schemas import (
    CropInputSchema,
    CropAgentResponse,
    ExtractedCropParameters,
)
from backend.services.crop_service import predict_crop

# Load environment variables
load_dotenv()


def create_predict_crop_tool() -> BaseTool:
    """
    Factory function creating a LangChain Tool wrapping the Crop ML service.
    
    Returns:
        A LangChain Tool callable by agents or pipelines.
    """
    @tool("predict_crop_tool", args_schema=CropInputSchema)
    def predict_crop_tool(
        nitrogen: float,
        phosphorus: float,
        potassium: float,
        temperature: float,
        humidity: float,
        ph: float,
        rainfall: float
    ) -> Dict[str, Any]:
        """
        Predicts the optimal crop to cultivate based on soil nutrients and climatic parameters.
        Requires:
        - nitrogen (0-300 kg/ha)
        - phosphorus (0-300 kg/ha)
        - potassium (0-300 kg/ha)
        - temperature (-10 to 60 °C)
        - humidity (0-100 %)
        - ph (0 to 14)
        - rainfall (0 to 1000 mm)
        """
        return predict_crop(
            nitrogen=nitrogen,
            phosphorus=phosphorus,
            potassium=potassium,
            temperature=temperature,
            humidity=humidity,
            ph=ph,
            rainfall=rainfall
        )

    return predict_crop_tool


class CropRecommendationAgent:
    """
    Agentic assistant specialized in interpreting farmer language,
    managing conversational context, and recommending optimal crops.
    """

    REQUIRED_FIELDS: List[str] = [
        "nitrogen",
        "phosphorus",
        "potassium",
        "temperature",
        "humidity",
        "ph",
        "rainfall",
    ]

    def __init__(self, llm: Optional[Any] = None):
        """
        Initialize the agent with optional LLM and create the ML tool.
        
        Args:
            llm: Optional LangChain chat model (ChatGroq, ChatOpenAI).
                 If omitted, initializes from environment variables or uses fallback parser.
        """
        self.tool = create_predict_crop_tool()
        self.llm = llm or self._initialize_llm()

    def _initialize_llm(self) -> Optional[Any]:
        """
        Initialize LLM with priority:
        1. Groq (Llama-3 — free, fast)
        2. OpenAI (GPT-4o-mini)
        3. None (graceful fallback to rule-based NLP extraction)
        """
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

    def extract_parameters_regex(self, text: str) -> Dict[str, Optional[float]]:
        """
        Robust rule-based entity extraction for soil & weather features from farmer text.
        Guarantees zero-failure parsing even when offline or LLM API keys are not supplied.
        """
        lower = text.lower()
        extracted: Dict[str, Optional[float]] = {field: None for field in self.REQUIRED_FIELDS}

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
            "temperature": [
                r"(?:temperature|temp)\s*(?:is|=|:|\bat\b)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:°?c|celsius)?",
                r"([0-9]+(?:\.[0-9]+)?)\s*°?c\s*(?:temperature|temp)?",
            ],
            "humidity": [
                r"(?:humidity)\s*(?:is|=|:|\bat\b)?\s*([0-9]+(?:\.[0-9]+)?)\s*%?",
                r"([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:humidity)?",
            ],
            "ph": [
                r"(?:ph|soil\s+ph)\s*(?:is|=|:|\blevel\b|\bof\b)?\s*([0-9]+(?:\.[0-9]+)?)",
                r"([0-9]+(?:\.[0-9]+)?)\s*(?:ph)",
            ],
            "rainfall": [
                r"(?:rainfall|rain)\s*(?:is|=|:|\bof\b)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:mm|cm)?",
                r"([0-9]+(?:\.[0-9]+)?)\s*(?:mm|cm)\s*(?:rainfall|rain)?",
            ],
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

        return extracted

    def extract_parameters(self, query: str) -> Dict[str, Optional[float]]:
        """
        Extract parameters using LLM structured output if available,
        falling back to deterministic regex parser.
        """
        if self.llm is not None:
            try:
                from langchain_core.prompts import ChatPromptTemplate
                structured_llm = self.llm.with_structured_output(ExtractedCropParameters)
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are an expert agricultural entity extraction assistant. Extract soil and climate numbers from the farmer's query. If a parameter is not mentioned, leave it as null."),
                    ("human", "{query}")
                ])
                chain = prompt | structured_llm
                result: ExtractedCropParameters = chain.invoke({"query": query})
                llm_dict = result.model_dump()
                # Verify non-null extractions
                if any(v is not None for v in llm_dict.values()):
                    return llm_dict
            except Exception:
                pass

        # Deterministic fallback
        return self.extract_parameters_regex(query)

    def identify_missing(self, params: Dict[str, Optional[float]]) -> List[str]:
        """Identify which mandatory crop features are absent."""
        return [field for field in self.REQUIRED_FIELDS if params.get(field) is None]

    def run(self, query: str) -> CropAgentResponse:
        """
        Process the farmer's natural language request.
        
        Args:
            query: The farmer's input text.
            
        Returns:
            CropAgentResponse with decision, confidence, and explanation.
        """
        extracted = self.extract_parameters(query)
        missing = self.identify_missing(extracted)

        # If data is incomplete, ask clarifying questions (Phase 2 requirement)
        if missing:
            missing_readable = ", ".join(m.replace("_", " ").title() for m in missing)
            clarifying_question = (
                f"I parsed your soil data, but I still need: {missing_readable} to give an accurate recommendation. "
                "Could you please share those values (or your location so our Weather Agent can look it up)?"
            )
            return CropAgentResponse(
                status="needs_clarification",
                recommended_crop=None,
                confidence=None,
                extracted_parameters=extracted,
                missing_parameters=missing,
                message=clarifying_question
            )

        # All parameters present: invoke LangChain ML tool
        try:
            validated_payload = {k: float(extracted[k]) for k in self.REQUIRED_FIELDS}  # type: ignore
            prediction_result = self.tool.invoke(validated_payload)

            crop = prediction_result["recommended_crop"]
            conf = float(prediction_result["confidence"])
            details = prediction_result["details"]

            farmer_message = (
                f"🌾 Recommendation: **{crop.title()}**\n"
                f"• Model Confidence: **{conf*100:.1f}%**\n"
                f"• Agronomic Reasoning: {details}\n"
                "• Next Step: Let's calculate your optimal fertilizer dosage and estimated yield!"
            )

            return CropAgentResponse(
                status="success",
                recommended_crop=crop,
                confidence=conf,
                extracted_parameters=extracted,
                missing_parameters=[],
                message=farmer_message
            )
        except Exception as exc:
            return CropAgentResponse(
                status="error",
                recommended_crop=None,
                confidence=None,
                extracted_parameters=extracted,
                missing_parameters=[],
                message=f"Error executing crop recommendation: {str(exc)}"
            )
