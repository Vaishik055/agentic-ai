"""
Multi-Agent Orchestrator (Week 8 / Phase 2 & 6).
Coordinates all autonomous agents (Weather, Soil, Market, Crop ML, Fertilizer ML,
Sustainability, Yield/Profit, Rotation, Reasoning) via a compiled LangGraph workflow.
"""

from typing import TypedDict, Optional, Dict, Any, List
from langgraph.graph import StateGraph, END

from backend.schemas.orchestrator_schemas import (
    OrchestratorQueryRequest,
    OrchestratorResponse,
)
from backend.agents.crop_recommendation_agent import CropRecommendationAgent
from backend.agents.fertilizer_agent import FertilizerAgent
from backend.agents.weather_agent import WeatherAgent
from backend.agents.soil_agent import SoilAgent
from backend.agents.market_agent import MarketAgent
from backend.agents.sustainability_agent import SustainabilityAgent
from backend.agents.yield_profit_agent import YieldProfitAgent
from backend.agents.rotation_planner_agent import RotationPlannerAgent
from backend.agents.reasoning_agent import ReasoningAgent


class FarmAdvisoryState(TypedDict, total=False):
    """Shared state container passed between all agents in the LangGraph pipeline."""
    query: str
    location: str
    land_area_acres: float
    irrigation_method: str
    extracted_params: Dict[str, Any]
    weather_data: Dict[str, Any]
    soil_data: Dict[str, Any]
    market_data: Dict[str, Any]
    crop_recommendation: Dict[str, Any]
    fertilizer_recommendation: Dict[str, Any]
    sustainability_report: Dict[str, Any]
    profitability_report: Dict[str, Any]
    rotation_plan: Dict[str, Any]
    explainable_reasoning: str
    missing_info: List[str]
    status: str


class FarmOrchestrator:
    """Master LangGraph supervisor orchestrating the end-to-end multi-agent pipeline."""

    def __init__(self):
        """Initialize all individual agents and build compiled LangGraph graph."""
        self.crop_agent = CropRecommendationAgent()
        self.fertilizer_agent = FertilizerAgent()
        self.weather_agent = WeatherAgent()
        self.soil_agent = SoilAgent()
        self.market_agent = MarketAgent()
        self.sustainability_agent = SustainabilityAgent()
        self.profit_agent = YieldProfitAgent()
        self.rotation_agent = RotationPlannerAgent()
        self.reasoning_agent = ReasoningAgent()

        self.graph = self._build_graph()

    def _node_entity_extraction(self, state: FarmAdvisoryState) -> Dict[str, Any]:
        """Phase 2: Extract entities and identify known parameters."""
        query = state.get("query", "")
        extracted = self.crop_agent.extract_parameters_regex(query)

        # Extract location if not explicitly provided
        loc = state.get("location")
        if not loc:
            loc = self.weather_agent.extract_location(query)

        return {
            "location": loc,
            "extracted_params": extracted
        }

    def _node_data_enrichment(self, state: FarmAdvisoryState) -> Dict[str, Any]:
        """Phase 3: Fetch autonomous weather, regional soil, and market price data."""
        loc = state.get("location", "Hyderabad")
        extracted = dict(state.get("extracted_params", {}))

        # Weather enrichment
        weather_res = self.weather_agent.run(f"Weather for {loc}")
        weather_dict = weather_res.model_dump()

        # If climate parameters missing in text, fill from Weather Agent
        if extracted.get("temperature") is None:
            extracted["temperature"] = weather_res.temperature
        if extracted.get("humidity") is None:
            extracted["humidity"] = weather_res.humidity
        if extracted.get("rainfall") is None:
            extracted["rainfall"] = weather_res.rainfall_forecast_mm

        # Soil profile enrichment
        soil_res = self.soil_agent.run(f"Soil in {loc}")
        soil_dict = soil_res.model_dump()

        if extracted.get("ph") is None:
            extracted["ph"] = soil_res.average_ph
        if extracted.get("nitrogen") is None:
            extracted["nitrogen"] = soil_res.average_nitrogen
        if extracted.get("phosphorus") is None:
            extracted["phosphorus"] = soil_res.average_phosphorus
        if extracted.get("potassium") is None:
            extracted["potassium"] = soil_res.average_potassium

        return {
            "weather_data": weather_dict,
            "soil_data": soil_dict,
            "extracted_params": extracted
        }

    def _node_recommendation(self, state: FarmAdvisoryState) -> Dict[str, Any]:
        """Phase 4: Run Crop ML and Fertilizer ML models."""
        params = state.get("extracted_params", {})
        soil_data = state.get("soil_data", {})

        # 1. Crop Recommendation ML
        validated_crop_input = {
            "nitrogen": float(params.get("nitrogen", 80.0)),
            "phosphorus": float(params.get("phosphorus", 40.0)),
            "potassium": float(params.get("potassium", 40.0)),
            "temperature": float(params.get("temperature", 25.0)),
            "humidity": float(params.get("humidity", 70.0)),
            "ph": float(params.get("ph", 6.5)),
            "rainfall": float(params.get("rainfall", 200.0)),
        }
        crop_res = self.crop_agent.tool.invoke(validated_crop_input)

        # 2. Fertilizer Recommendation ML
        soil_type = soil_data.get("dominant_soil_type", "Loamy")
        fert_res = self.fertilizer_agent.tool.invoke({
            "nitrogen": validated_crop_input["nitrogen"],
            "phosphorus": validated_crop_input["phosphorus"],
            "potassium": validated_crop_input["potassium"],
            "soil_type": soil_type,
            "crop_type": crop_res["recommended_crop"].title()
        })

        # Market price lookup for recommended crop
        mkt_res = self.market_agent.run(f"{crop_res['recommended_crop']} in Telangana")

        return {
            "crop_recommendation": crop_res,
            "fertilizer_recommendation": fert_res,
            "market_data": mkt_res.model_dump()
        }

    def _node_evaluation(self, state: FarmAdvisoryState) -> Dict[str, Any]:
        """Phase 5: Run Sustainability, Profitability, and Rotation evaluations."""
        crop_name = state.get("crop_recommendation", {}).get("recommended_crop", "Rice")
        fert_name = state.get("fertilizer_recommendation", {}).get("recommended_fertilizer", "Urea")
        acres = float(state.get("land_area_acres", 2.0))
        irr = state.get("irrigation_method", "Flood")
        soil_type = state.get("soil_data", {}).get("dominant_soil_type", "Black")
        mkt_price = float(state.get("market_data", {}).get("modal_price_per_quintal", 2350.0))

        # Sustainability
        from backend.schemas.evaluation_schemas import (
            SustainabilityRequest,
            ProfitabilityRequest,
            RotationPlanRequest,
        )
        sust_res = self.sustainability_agent.evaluate(SustainabilityRequest(
            crop=crop_name,
            irrigation_method=irr,
            fertilizer_used=fert_name,
            soil_organic_carbon=0.55,
            rainfall_mm=200.0
        ))

        # Profitability
        profit_res = self.profit_agent.evaluate(ProfitabilityRequest(
            crop=crop_name,
            land_area_acres=acres,
            mandi_price_per_quintal=mkt_price
        ))

        # Rotation
        rot_res = self.rotation_agent.plan_rotation(RotationPlanRequest(
            current_crop=crop_name,
            soil_type=soil_type,
            irrigation_availability="Assured"
        ))

        return {
            "sustainability_report": sust_res.model_dump(),
            "profitability_report": profit_res.model_dump(),
            "rotation_plan": rot_res.model_dump()
        }

    def _node_reasoning_synthesis(self, state: FarmAdvisoryState) -> Dict[str, Any]:
        """Phase 6: Synthesize explainable recommendation package."""
        reasoning_text = self.reasoning_agent.synthesize(dict(state))
        return {
            "explainable_reasoning": reasoning_text,
            "status": "success"
        }

    def _build_graph(self) -> Any:
        """Construct and compile the LangGraph StateGraph."""
        builder = StateGraph(FarmAdvisoryState)

        builder.add_node("extract_entities", self._node_entity_extraction)
        builder.add_node("enrich_data", self._node_data_enrichment)
        builder.add_node("recommend_crop_and_fert", self._node_recommendation)
        builder.add_node("evaluate_farm_plan", self._node_evaluation)
        builder.add_node("synthesize_reasoning", self._node_reasoning_synthesis)

        builder.set_entry_point("extract_entities")
        builder.add_edge("extract_entities", "enrich_data")
        builder.add_edge("enrich_data", "recommend_crop_and_fert")
        builder.add_edge("recommend_crop_and_fert", "evaluate_farm_plan")
        builder.add_edge("evaluate_farm_plan", "synthesize_reasoning")
        builder.add_edge("synthesize_reasoning", END)

        return builder.compile()

    def process_query(self, req: OrchestratorQueryRequest) -> OrchestratorResponse:
        """
        Execute full multi-agent pipeline and return comprehensive advice.
        """
        initial_state: FarmAdvisoryState = {
            "query": req.query,
            "location": req.location or "",
            "land_area_acres": req.land_area_acres or 2.0,
            "irrigation_method": req.irrigation_method or "Flood",
            "missing_info": [],
            "status": "in_progress"
        }

        final_state = self.graph.invoke(initial_state)

        return OrchestratorResponse(
            status=final_state.get("status", "success"),
            primary_recommendation={
                "crop": final_state.get("crop_recommendation", {}),
                "fertilizer": final_state.get("fertilizer_recommendation", {})
            },
            environmental_context={
                "weather": final_state.get("weather_data", {}),
                "soil": final_state.get("soil_data", {})
            },
            market_and_profitability={
                "market": final_state.get("market_data", {}),
                "profitability": final_state.get("profitability_report", {})
            },
            sustainability_and_planning={
                "sustainability": final_state.get("sustainability_report", {}),
                "rotation": final_state.get("rotation_plan", {})
            },
            explainable_reasoning=final_state.get("explainable_reasoning", "No synthesis generated."),
            missing_information=[]
        )


_orchestrator_instance = None


def get_farm_orchestrator() -> FarmOrchestrator:
    """Get or initialize singleton instance of FarmOrchestrator."""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = FarmOrchestrator()
    return _orchestrator_instance
