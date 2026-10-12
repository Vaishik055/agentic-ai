"""
FastAPI application for Sustainable Farming Advisory System.
Central API Gateway exposing individual specialized agents and the master
LangGraph multi-agent orchestrator.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas.crop_schemas import (
    CropInputSchema,
    CropPredictionResponse,
    CropAgentQueryRequest,
    CropAgentResponse,
)
from backend.schemas.fertilizer_schemas import (
    FertilizerInputSchema,
    FertilizerPredictionResponse,
    FertilizerAgentQueryRequest,
    FertilizerAgentResponse,
)
from backend.schemas.data_schemas import (
    WeatherAgentRequest,
    WeatherDataResponse,
    SoilAgentRequest,
    SoilProfileResponse,
    MarketAgentRequest,
    MarketPriceResponse,
)
from backend.schemas.rag_schemas import (
    KnowledgeQueryRequest,
    KnowledgeAgentResponse,
)
from backend.schemas.evaluation_schemas import (
    SustainabilityRequest,
    SustainabilityResponse,
    ProfitabilityRequest,
    ProfitabilityResponse,
    RotationPlanRequest,
    RotationPlanResponse,
)
from backend.schemas.orchestrator_schemas import (
    OrchestratorQueryRequest,
    OrchestratorResponse,
)
from backend.services.crop_service import get_crop_service
from backend.services.fertilizer_service import get_fertilizer_service
from backend.agents.crop_recommendation_agent import CropRecommendationAgent
from backend.agents.fertilizer_agent import FertilizerAgent
from backend.agents.weather_agent import WeatherAgent
from backend.agents.soil_agent import SoilAgent
from backend.agents.market_agent import MarketAgent
from backend.agents.knowledge_agent import KnowledgeAgent
from backend.agents.sustainability_agent import SustainabilityAgent
from backend.agents.yield_profit_agent import YieldProfitAgent
from backend.agents.rotation_planner_agent import RotationPlannerAgent
from backend.agents.orchestrator import get_farm_orchestrator

app = FastAPI(
    title="Sustainable Farming AI Assistant API",
    description="Agentic advisory system for personalized, sustainable crop, fertilizer, climate, and farm economics management.",
    version="1.5.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton agent instances
_crop_agent_instance = None
_fertilizer_agent_instance = None
_weather_agent_instance = None
_soil_agent_instance = None
_market_agent_instance = None
_knowledge_agent_instance = None
_sustainability_agent_instance = None
_profitability_agent_instance = None
_rotation_agent_instance = None


def get_crop_agent() -> CropRecommendationAgent:
    global _crop_agent_instance
    if _crop_agent_instance is None:
        _crop_agent_instance = CropRecommendationAgent()
    return _crop_agent_instance


def get_fertilizer_agent() -> FertilizerAgent:
    global _fertilizer_agent_instance
    if _fertilizer_agent_instance is None:
        _fertilizer_agent_instance = FertilizerAgent()
    return _fertilizer_agent_instance


def get_weather_agent() -> WeatherAgent:
    global _weather_agent_instance
    if _weather_agent_instance is None:
        _weather_agent_instance = WeatherAgent()
    return _weather_agent_instance


def get_soil_agent() -> SoilAgent:
    global _soil_agent_instance
    if _soil_agent_instance is None:
        _soil_agent_instance = SoilAgent()
    return _soil_agent_instance


def get_market_agent() -> MarketAgent:
    global _market_agent_instance
    if _market_agent_instance is None:
        _market_agent_instance = MarketAgent()
    return _market_agent_instance


def get_knowledge_agent() -> KnowledgeAgent:
    global _knowledge_agent_instance
    if _knowledge_agent_instance is None:
        _knowledge_agent_instance = KnowledgeAgent()
    return _knowledge_agent_instance


def get_sustainability_agent() -> SustainabilityAgent:
    global _sustainability_agent_instance
    if _sustainability_agent_instance is None:
        _sustainability_agent_instance = SustainabilityAgent()
    return _sustainability_agent_instance


def get_profitability_agent() -> YieldProfitAgent:
    global _profitability_agent_instance
    if _profitability_agent_instance is None:
        _profitability_agent_instance = YieldProfitAgent()
    return _profitability_agent_instance


def get_rotation_agent() -> RotationPlannerAgent:
    global _rotation_agent_instance
    if _rotation_agent_instance is None:
        _rotation_agent_instance = RotationPlannerAgent()
    return _rotation_agent_instance


@app.get("/", tags=["Health"])
def root():
    """Root health and status endpoint."""
    return {
        "status": "healthy",
        "service": "Sustainable Farming AI API",
        "version": "1.5.0"
    }


@app.get("/health", tags=["Health"])
def health_check():
    """Detailed health check validating ML services and agents availability."""
    try:
        crop_service = get_crop_service()
        fertilizer_service = get_fertilizer_service()
        crops_loaded = crop_service.model is not None
        fertilizer_loaded = fertilizer_service.model is not None

        is_healthy = crops_loaded and fertilizer_loaded
        return {
            "status": "healthy" if is_healthy else "degraded",
            "crop_model_loaded": crops_loaded,
            "fertilizer_model_loaded": fertilizer_loaded,
            "rag_indexed": True,
            "orchestrator_ready": True
        }
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service initialization error: {str(exc)}"
        )


# ============================================================================
# Master Orchestration Endpoint (Week 8 / Phase 6)
# ============================================================================

@app.post(
    "/agent/orchestrate",
    response_model=OrchestratorResponse,
    tags=["Master Multi-Agent Orchestrator (Week 8)"],
    summary="End-to-end multi-agent orchestration for farmer advisory"
)
def orchestrate_farm_advisory(payload: OrchestratorQueryRequest):
    """
    Master endpoint coordinating:
    - Weather Agent (OpenWeather / Regional Climate)
    - Soil Agent (ICAR Soil Atlas)
    - Market Price Agent (Agmarknet APMC)
    - Crop Recommendation Agent (Random Forest ML)
    - Fertilizer Agent (Nutrient Deficit & Dosage)
    - Sustainability Agent (0-100 Eco Index)
    - Yield & Profitability Agent (ROI & Economics)
    - Crop Rotation Planner (Nitrogen Restoration)
    - Reasoning Agent (Phase 6 Explainable Synthesis)
    """
    try:
        orchestrator = get_farm_orchestrator()
        return orchestrator.process_query(payload)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Orchestration pipeline failure: {str(exc)}"
        )


# ============================================================================
# Individual Agent Endpoints
# ============================================================================

@app.post("/predict/crop", response_model=CropPredictionResponse, tags=["Crop ML Model (Week 2)"])
def predict_crop_endpoint(input_data: CropInputSchema):
    try:
        service = get_crop_service()
        return service.predict(input_data)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@app.post("/agent/crop", response_model=CropAgentResponse, tags=["Crop AI Agent (Week 3)"])
def agent_crop_endpoint(payload: CropAgentQueryRequest):
    try:
        agent = get_crop_agent()
        return agent.run(payload.query)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@app.post("/predict/fertilizer", response_model=FertilizerPredictionResponse, tags=["Fertilizer ML Model (Week 4)"])
def predict_fertilizer_endpoint(input_data: FertilizerInputSchema):
    try:
        service = get_fertilizer_service()
        return service.predict(input_data)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@app.post("/agent/fertilizer", response_model=FertilizerAgentResponse, tags=["Fertilizer AI Agent (Week 4)"])
def agent_fertilizer_endpoint(payload: FertilizerAgentQueryRequest):
    try:
        agent = get_fertilizer_agent()
        return agent.run(payload.query)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@app.post("/agent/weather", response_model=WeatherDataResponse, tags=["Data Agents (Week 5)"])
def agent_weather_endpoint(payload: WeatherAgentRequest):
    try:
        agent = get_weather_agent()
        return agent.run(payload.location)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@app.post("/agent/soil", response_model=SoilProfileResponse, tags=["Data Agents (Week 5)"])
def agent_soil_endpoint(payload: SoilAgentRequest):
    try:
        agent = get_soil_agent()
        return agent.run(payload.location)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@app.post("/agent/market", response_model=MarketPriceResponse, tags=["Data Agents (Week 5)"])
def agent_market_endpoint(payload: MarketAgentRequest):
    try:
        agent = get_market_agent()
        return agent.run(f"{payload.crop} in {payload.state or 'Telangana'}")
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@app.post("/agent/knowledge", response_model=KnowledgeAgentResponse, tags=["Knowledge Agent (Week 6)"])
def agent_knowledge_endpoint(payload: KnowledgeQueryRequest):
    try:
        agent = get_knowledge_agent()
        return agent.run(payload.query)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@app.post("/agent/sustainability", response_model=SustainabilityResponse, tags=["Evaluation Agents (Week 7)"])
def agent_sustainability_endpoint(payload: SustainabilityRequest):
    try:
        agent = get_sustainability_agent()
        return agent.evaluate(payload)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@app.post("/agent/profitability", response_model=ProfitabilityResponse, tags=["Evaluation Agents (Week 7)"])
def agent_profitability_endpoint(payload: ProfitabilityRequest):
    try:
        agent = get_profitability_agent()
        return agent.evaluate(payload)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@app.post("/agent/rotation", response_model=RotationPlanResponse, tags=["Evaluation Agents (Week 7)"])
def agent_rotation_endpoint(payload: RotationPlanRequest):
    try:
        agent = get_rotation_agent()
        return agent.plan_rotation(payload)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
