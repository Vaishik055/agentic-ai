"""
Pydantic schemas for Crop Recommendation service and agent.
Defines strict validation rules for soil, weather input features, and agent queries.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class CropInputSchema(BaseModel):
    """Input features required for crop prediction with biological/physical bounds."""
    nitrogen: float = Field(
        ...,
        ge=0.0,
        le=300.0,
        description="Nitrogen content in soil (kg/ha), typically 0 to 300."
    )
    phosphorus: float = Field(
        ...,
        ge=0.0,
        le=300.0,
        description="Phosphorus content in soil (kg/ha), typically 0 to 300."
    )
    potassium: float = Field(
        ...,
        ge=0.0,
        le=300.0,
        description="Potassium content in soil (kg/ha), typically 0 to 300."
    )
    temperature: float = Field(
        ...,
        ge=-10.0,
        le=60.0,
        description="Ambient temperature in degrees Celsius (°C)."
    )
    humidity: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Relative air humidity percentage (0-100%)."
    )
    ph: float = Field(
        ...,
        ge=0.0,
        le=14.0,
        description="Soil pH level on standard scale (0.0 to 14.0)."
    )
    rainfall: float = Field(
        ...,
        ge=0.0,
        le=1000.0,
        description="Annual or seasonal rainfall in millimeters (mm)."
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "nitrogen": 90.0,
                "phosphorus": 42.0,
                "potassium": 43.0,
                "temperature": 25.0,
                "humidity": 80.0,
                "ph": 6.5,
                "rainfall": 200.0
            }
        }
    }


class CropPredictionResponse(BaseModel):
    """Response returned after running model inference."""
    recommended_crop: str = Field(..., description="The predicted optimal crop name.")
    confidence: float = Field(..., description="Model confidence score between 0.0 and 1.0.")
    details: str = Field(..., description="Human-readable explanation of why this crop fits.")


class CropAgentQueryRequest(BaseModel):
    """Farmer's natural language query payload for the Crop Recommendation Agent."""
    query: str = Field(
        ...,
        description="Unstructured natural language query from the farmer containing soil and weather details.",
        examples=["My soil has 90 nitrogen, 42 phosphorus, 43 potassium, temperature is 25°C, humidity 80%, pH 6.5, rainfall 200mm. What should I grow?"]
    )


class ExtractedCropParameters(BaseModel):
    """Structured parameters extracted from natural language farmer text."""
    nitrogen: Optional[float] = None
    phosphorus: Optional[float] = None
    potassium: Optional[float] = None
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    ph: Optional[float] = None
    rainfall: Optional[float] = None


class CropAgentResponse(BaseModel):
    """Structured response from the Crop Recommendation Agent."""
    status: str = Field(..., description="'success' if prediction made, 'needs_clarification' if parameters missing.")
    recommended_crop: Optional[str] = Field(None, description="Recommended crop name.")
    confidence: Optional[float] = Field(None, description="Prediction confidence score.")
    extracted_parameters: Dict[str, Optional[float]] = Field(..., description="Parameters extracted from text.")
    missing_parameters: List[str] = Field(default_factory=list, description="List of required parameters that are missing.")
    message: str = Field(..., description="Conversational explanation or clarifying question for the farmer.")
