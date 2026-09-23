"""
FastAPI application for Sustainable Farming Advisory System.
Provides health endpoints, direct ML inference, and Agentic query endpoints
for both Crop Recommendation and Fertilizer Advisory.
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
from backend.services.crop_service import get_crop_service
from backend.services.fertilizer_service import get_fertilizer_service
from backend.agents.crop_recommendation_agent import CropRecommendationAgent
from backend.agents.fertilizer_agent import FertilizerAgent

app = FastAPI(
    title="Sustainable Farming AI Assistant API",
    description="Agentic advisory system for personalized, sustainable crop and fertilizer management.",
    version="1.1.0"
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


def get_crop_agent() -> CropRecommendationAgent:
    """Get or initialize singleton instance of CropRecommendationAgent."""
    global _crop_agent_instance
    if _crop_agent_instance is None:
        _crop_agent_instance = CropRecommendationAgent()
    return _crop_agent_instance


def get_fertilizer_agent() -> FertilizerAgent:
    """Get or initialize singleton instance of FertilizerAgent."""
    global _fertilizer_agent_instance
    if _fertilizer_agent_instance is None:
        _fertilizer_agent_instance = FertilizerAgent()
    return _fertilizer_agent_instance


@app.get("/", tags=["Health"])
def root():
    """Root health and status endpoint."""
    return {
        "status": "healthy",
        "service": "Sustainable Farming AI API",
        "version": "1.1.0"
    }


@app.get("/health", tags=["Health"])
def health_check():
    """Detailed health check validating ML services availability."""
    try:
        crop_service = get_crop_service()
        fertilizer_service = get_fertilizer_service()
        crops_loaded = crop_service.model is not None
        fertilizer_loaded = fertilizer_service.model is not None

        is_healthy = crops_loaded and fertilizer_loaded
        return {
            "status": "healthy" if is_healthy else "degraded",
            "crop_model_loaded": crops_loaded,
            "fertilizer_model_loaded": fertilizer_loaded
        }
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service initialization error: {str(exc)}"
        )


# ============================================================================
# Week 2 & Week 3: Crop Recommendation Endpoints
# ============================================================================

@app.post(
    "/predict/crop",
    response_model=CropPredictionResponse,
    tags=["Crop ML Model (Week 2)"],
    summary="Direct ML prediction from structured soil & climate JSON"
)
def predict_crop_endpoint(input_data: CropInputSchema):
    """
    Direct Week 2 ML inference endpoint.
    Receives JSON with N, P, K, temperature, humidity, pH, and rainfall.
    """
    try:
        service = get_crop_service()
        return service.predict(input_data)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Prediction error: {str(exc)}"
        )


@app.post(
    "/agent/crop",
    response_model=CropAgentResponse,
    tags=["Crop AI Agent (Week 3)"],
    summary="Agentic crop recommendation from natural language farmer text"
)
def agent_crop_endpoint(payload: CropAgentQueryRequest):
    """
    Week 3 Agentic endpoint:
    Accepts natural language from the farmer, extracts entities,
    invokes the LangChain Crop ML tool, asks clarifying questions if data is missing,
    and returns an explainable recommendation.
    """
    try:
        agent = get_crop_agent()
        return agent.run(payload.query)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent execution failure: {str(exc)}"
        )


# ============================================================================
# Week 4: Fertilizer Recommendation Endpoints
# ============================================================================

@app.post(
    "/predict/fertilizer",
    response_model=FertilizerPredictionResponse,
    tags=["Fertilizer ML Model (Week 4)"],
    summary="Direct ML fertilizer prediction from soil, crop, and nutrient JSON"
)
def predict_fertilizer_endpoint(input_data: FertilizerInputSchema):
    """
    Direct Week 4 ML inference endpoint.
    Receives soil NPK, moisture, soil type, and target crop to calculate fertilizer & dosage.
    """
    try:
        service = get_fertilizer_service()
        return service.predict(input_data)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Fertilizer prediction error: {str(exc)}"
        )


@app.post(
    "/agent/fertilizer",
    response_model=FertilizerAgentResponse,
    tags=["Fertilizer AI Agent (Week 4)"],
    summary="Agentic fertilizer recommendation from natural language farmer text"
)
def agent_fertilizer_endpoint(payload: FertilizerAgentQueryRequest):
    """
    Week 4 Agentic endpoint:
    Accepts natural language queries about fertilizer, extracts NPK and crop details,
    invokes the LangChain fertilizer tool, and returns dosage and eco-friendly tips.
    """
    try:
        agent = get_fertilizer_agent()
        return agent.run(payload.query)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Fertilizer agent execution failure: {str(exc)}"
        )
