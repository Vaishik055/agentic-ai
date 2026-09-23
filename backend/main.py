"""
FastAPI application for Sustainable Farming Advisory System.
Provides health endpoints, direct ML inference, and Agentic query endpoints.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from backend.schemas.crop_schemas import (
    CropInputSchema,
    CropPredictionResponse,
    CropAgentQueryRequest,
    CropAgentResponse,
)
from backend.services.crop_service import get_crop_service
from backend.agents.crop_recommendation_agent import CropRecommendationAgent

app = FastAPI(
    title="Sustainable Farming AI Assistant API",
    description="Agentic advisory system for personalized, sustainable crop and farm management.",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton agent instance
_crop_agent_instance = None


def get_crop_agent() -> CropRecommendationAgent:
    """Get or initialize singleton instance of CropRecommendationAgent."""
    global _crop_agent_instance
    if _crop_agent_instance is None:
        _crop_agent_instance = CropRecommendationAgent()
    return _crop_agent_instance


@app.get("/", tags=["Health"])
def root():
    """Root health and status endpoint."""
    return {
        "status": "healthy",
        "service": "Sustainable Farming AI API",
        "version": "1.0.0"
    }


@app.get("/health", tags=["Health"])
def health_check():
    """Detailed health check validating ML service availability."""
    try:
        service = get_crop_service()
        is_ready = service.model is not None
        return {
            "status": "healthy" if is_ready else "degraded",
            "model_loaded": is_ready
        }
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service initialization error: {str(exc)}"
        )


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
