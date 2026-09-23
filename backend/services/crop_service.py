"""
Crop Recommendation Service.
Loads the trained Random Forest model, scaler, and label encoder to perform
validated, high-confidence crop inference.
"""

from pathlib import Path
from typing import Dict, Any, Union
import joblib
import numpy as np
import pandas as pd

from backend.schemas.crop_schemas import CropInputSchema, CropPredictionResponse


class CropRecommendationService:
    """Production service class wrapping the Random Forest Crop Recommendation Model."""

    def __init__(self, base_dir: Union[Path, str, None] = None):
        """Initialize and load serialized ML pipeline artifacts."""
        if base_dir is None:
            self.base_dir = Path(__file__).resolve().parent.parent
        else:
            self.base_dir = Path(base_dir)

        model_path = self.base_dir / "models" / "crop_model.pkl"
        scaler_path = self.base_dir / "scalers" / "crop_scaler.pkl"
        encoder_path = self.base_dir / "encoders" / "label_encoder.pkl"

        if not model_path.exists() or not scaler_path.exists() or not encoder_path.exists():
            raise FileNotFoundError(
                f"Missing ML artifacts in {self.base_dir}. Ensure model, scaler, and encoder files exist."
            )

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.label_encoder = joblib.load(encoder_path)
        self.feature_names = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

    def predict(self, input_data: Union[CropInputSchema, Dict[str, Any]]) -> CropPredictionResponse:
        """
        Validate inputs and predict the optimal crop label with confidence.
        
        Args:
            input_data: Validated CropInputSchema object or dictionary of feature values.
            
        Returns:
            CropPredictionResponse containing recommended crop, confidence, and summary.
        """
        if isinstance(input_data, dict):
            validated_input = CropInputSchema(**input_data)
        else:
            validated_input = input_data

        features_df = pd.DataFrame(
            [[
                validated_input.nitrogen,
                validated_input.phosphorus,
                validated_input.potassium,
                validated_input.temperature,
                validated_input.humidity,
                validated_input.ph,
                validated_input.rainfall
            ]],
            columns=self.feature_names
        )

        scaled_features = self.scaler.transform(features_df)
        prediction_idx = self.model.predict(scaled_features)[0]
        crop_label = str(self.label_encoder.inverse_transform([prediction_idx])[0])

        probabilities = self.model.predict_proba(scaled_features)[0]
        confidence = float(np.max(probabilities))

        explanation = (
            f"Recommended {crop_label.title()} with {confidence*100:.1f}% confidence based on "
            f"soil N-P-K ({validated_input.nitrogen}-{validated_input.phosphorus}-{validated_input.potassium}), "
            f"pH {validated_input.ph}, temperature {validated_input.temperature}°C, "
            f"humidity {validated_input.humidity}%, and rainfall {validated_input.rainfall}mm."
        )

        return CropPredictionResponse(
            recommended_crop=crop_label,
            confidence=round(confidence, 4),
            details=explanation
        )


# Global singleton instance for efficient reuse across endpoints & agents
_service_instance = None


def get_crop_service() -> CropRecommendationService:
    """Get or create singleton instance of CropRecommendationService."""
    global _service_instance
    if _service_instance is None:
        _service_instance = CropRecommendationService()
    return _service_instance


def predict_crop(
    nitrogen: float,
    phosphorus: float,
    potassium: float,
    temperature: float,
    humidity: float,
    ph: float,
    rainfall: float
) -> Dict[str, Any]:
    """
    Convenience function to run crop prediction with individual parameters.
    """
    service = get_crop_service()
    data = CropInputSchema(
        nitrogen=nitrogen,
        phosphorus=phosphorus,
        potassium=potassium,
        temperature=temperature,
        humidity=humidity,
        ph=ph,
        rainfall=rainfall
    )
    result = service.predict(data)
    return result.model_dump()
