"""
Unit tests for CropRecommendationService (Week 2 baseline).
"""

import pytest
from backend.services.crop_service import CropRecommendationService, predict_crop
from backend.schemas.crop_schemas import CropInputSchema


def test_crop_service_initialization():
    service = CropRecommendationService()
    assert service.model is not None
    assert service.scaler is not None
    assert service.label_encoder is not None


def test_predict_crop_rice():
    result = predict_crop(
        nitrogen=90.0,
        phosphorus=42.0,
        potassium=43.0,
        temperature=25.0,
        humidity=80.0,
        ph=6.5,
        rainfall=200.0
    )
    assert result["recommended_crop"] == "rice"
    assert result["confidence"] > 0.8
    assert "Rice" in result["details"]


def test_invalid_input_ranges():
    with pytest.raises(Exception):
        CropInputSchema(
            nitrogen=500.0,  # Exceeds max limit 300
            phosphorus=42.0,
            potassium=43.0,
            temperature=25.0,
            humidity=80.0,
            ph=6.5,
            rainfall=200.0
        )
