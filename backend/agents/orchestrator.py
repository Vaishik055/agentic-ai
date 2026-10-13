"""
Multi-Agent Orchestrator (Week 8 / Phase 2 & 6).
Coordinates all autonomous agents (Weather, Soil, Market, Crop ML, Fertilizer ML,
Sustainability, Yield/Profit, Rotation, Reasoning) via a compiled LangGraph workflow.
"""

from typing import TypedDict, Optional, Dict, Any, List

try:
    from langgraph.graph import StateGraph, END
except ImportError:
    StateGraph = None
    END = "__end__"

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
    """Shared state dictionary passed between all agent nodes in LangGraph."""
    query: str
    location: str
    land_area_acres: float
    irrigation_method: str
    extracted_features: Dict[str, Any]
    weather_data: Dict[str, Any]
    soil_data: Dict[str, Any]
    market_data: Dict[str, Any]
    crop_recommendation: Dict[str, Any]
    fertilizer_recommendation: Dict[str, Any]
    profitability_report: Dict[str, Any]
    sustainability_report: Dict[str, Any]
    rotation_plan: Dict[str, Any]
    explainable_reasoning: str
    missing_info: List[str]
    status: str


class FarmOrchestrator:
    """Master LangGraph supervisor orchestrating all 8 autonomous agronomic agents."""

    def __init__(self):
        """Initialize all sub-agents and build the compiled StateGraph."""
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
        """Node 1: Extract soil and climatic parameters from farmer's natural text query."""
        query = state.get("query", "")
        extracted, missing = self.crop_agent.extract_parameters_from_text(query)
        return {
            "extracted_features": extracted,
            "missing_info": missing
        }

    def _node_data_enrichment(self, state: FarmAdvisoryState) -> Dict[str, Any]:
        """Node 2: Autonomously query Weather and Soil agents to enrich missing data points."""
        extracted = dict(state.get("extracted_features", {}))
        missing = list(state.get("missing_info", []))
        location = state.get("location") or "Hyderabad, Telangana"

        # 1. Fetch live agro-weather
        weather_res = self.weather_agent.run(location)
        weather_data = weather_res.model_dump()

        if "temperature" in missing:
            extracted["temperature"] = weather_res.temperature
            missing.remove("temperature")
        if "humidity" in missing:
            extracted["humidity"] = weather_res.humidity
            missing.remove("humidity")
        if "rainfall" in missing:
            extracted["rainfall"] = weather_res.rainfall_mm
            missing.remove("rainfall")

        # 2. Fetch regional soil atlas
        soil_res = self.soil_agent.run(location)
        soil_data = soil_res.model_dump()

        if "ph" in missing:
            extracted["ph"] = soil_res.ph
            missing.remove("ph")

        return {
            "extracted_features": extracted,
            "missing_info": missing,
            "weather_data": weather_data,
            "soil_data": soil_data
        }

    def _node_recommendation(self, state: FarmAdvisoryState) -> Dict[str, Any]:
        """Node 3: Execute Crop ML classifier and Fertilizer optimization engine."""
        extracted = state.get("extracted_features", {})
        query = state.get("query", "")

        # 1. Crop ML Recommendation
        crop_res = self.crop_agent.run(query)
        crop_data = crop_res.model_dump()
        rec_crop = crop_res.recommended_crop or "rice"

        # 2. Fertilizer ML Recommendation
        fert_query = (
            f"I want to grow {rec_crop} with soil nitrogen {extracted.get('N', 50)}, "
            f"phosphorus {extracted.get('P', 50)}, potassium {extracted.get('K', 50)}."
        )
        fert_res = self.fertilizer_agent.run(fert_query)
        fert_data = fert_res.model_dump()

        # 3. Market APMC Mandi Data
        market_res = self.market_agent.run(crop=rec_crop, mandi="Hyderabad")
        market_data = market_res.model_dump()

        return {
            "crop_recommendation": crop_data,
            "fertilizer_recommendation": fert_data,
            "market_data": market_data
        }

    def _node_evaluation(self, state: FarmAdvisoryState) -> Dict[str, Any]:
        """Node 4: Evaluate Economics (Yield/Profit), Sustainability (0-100), and Crop Rotation."""
        crop_rec = state.get("crop_recommendation", {})
        rec_crop = crop_rec.get("recommended_crop", "Rice")
        acres = state.get("land_area_acres", 2.0)
        irr_method = state.get("irrigation_method", "Flood")
        fert_rec = state.get("fertilizer_recommendation", {})
        fert_name = fert_rec.get("recommended_fertilizer", "Urea")
        weather = state.get("weather_data", {})
        rain = weather.get("rainfall_mm", 150.0)

        # 1. Yield and Profitability
        profit_res = self.profit_agent.calculate(
            crop_name=rec_crop,
            land_area_acres=acres
        )

        # 2. Sustainability Index (0-100)
        from backend.schemas.evaluation_schemas import SustainabilityRequest
        sus_req = SustainabilityRequest(
            crop=rec_crop,
            irrigation_method=irr_method,
            fertilizer_used=fert_name,
            soil_organic_carbon=0.65,
            rainfall_mm=rain,
            residue_burning=False,
            organic_amendments=True,
            cover_cropping=True
        )
        sus_res = self.sustainability_agent.evaluate(sus_req)

        # 3. Crop Rotation Plan
        rotation_res = self.rotation_agent.plan(
            primary_crop=rec_crop,
            current_season="Kharif"
        )

        return {
            "profitability_report": profit_res.model_dump(),
            "sustainability_report": sus_res.model_dump(),
            "rotation_plan": rotation_res.model_dump()
        }

    def _node_reasoning_synthesis(self, state: FarmAdvisoryState) -> Dict[str, Any]:
        """Node 5: Phase 6 Reasoning Engine formulating explainable, transparent farmer advice."""
        reasoning_text = self.reasoning_agent.synthesize_farm_advice(
            crop_data=state.get("crop_recommendation", {}),
            fert_data=state.get("fertilizer_recommendation", {}),
            weather_data=state.get("weather_data", {}),
            soil_data=state.get("soil_data", {}),
            profit_data=state.get("profitability_report", {}),
            sus_data=state.get("sustainability_report", {}),
            rotation_data=state.get("rotation_plan", {})
        )

        return {
            "explainable_reasoning": reasoning_text,
            "status": "success"
        }

    def _build_graph(self) -> Any:
        """Construct and compile the LangGraph StateGraph, or provide sequential fallback."""
        if StateGraph is not None:
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

        # Pure Python sequential fallback if langgraph is not installed
        class _SequentialWorkflow:
            def __init__(self, orch):
                self.orch = orch

            def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
                s = dict(state)
                s.update(self.orch._node_entity_extraction(s))
                s.update(self.orch._node_data_enrichment(s))
                s.update(self.orch._node_recommendation(s))
                s.update(self.orch._node_evaluation(s))
                s.update(self.orch._node_reasoning_synthesis(s))
                return s

        return _SequentialWorkflow(self)

    def process_query(self, req: OrchestratorQueryRequest) -> OrchestratorResponse:
        """Execute full multi-agent pipeline and return comprehensive advice."""
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