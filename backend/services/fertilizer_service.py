"""
Fertilizer Recommendation Service.
Provides ML inference for fertilizer prediction, nutrient deficit analysis,
dosage calculation, application scheduling, and sustainable soil practices.
"""

from pathlib import Path
from typing import Dict, Any, Union, List
import joblib
import numpy as np
import pandas as pd

from backend.schemas.fertilizer_schemas import (
    FertilizerInputSchema,
    FertilizerPredictionResponse,
    NutrientDeficitReport,
)


class FertilizerRecommendationService:
    """Production service evaluating soil nutrition and recommending targeted fertilizers."""

    # Agronomic standard N-P-K thresholds (kg/ha) for major crop types
    CROP_NUTRIENT_BENCHMARKS = {
        "rice": {"N": 80.0, "P": 40.0, "K": 40.0, "default_dose": 45.0},
        "maize": {"N": 100.0, "P": 50.0, "K": 30.0, "default_dose": 50.0},
        "wheat": {"N": 100.0, "P": 50.0, "K": 40.0, "default_dose": 50.0},
        "cotton": {"N": 120.0, "P": 45.0, "K": 45.0, "default_dose": 55.0},
        "sugarcane": {"N": 150.0, "P": 60.0, "K": 60.0, "default_dose": 75.0},
        "chickpea": {"N": 25.0, "P": 60.0, "K": 30.0, "default_dose": 35.0},
        "kidneybeans": {"N": 30.0, "P": 65.0, "K": 25.0, "default_dose": 35.0},
        "pigeonpeas": {"N": 25.0, "P": 60.0, "K": 25.0, "default_dose": 35.0},
        "ground nuts": {"N": 25.0, "P": 50.0, "K": 40.0, "default_dose": 40.0},
        "default": {"N": 70.0, "P": 45.0, "K": 40.0, "default_dose": 45.0},
    }

    def __init__(self, base_dir: Union[Path, str, None] = None):
        """Initialize and load serialized ML models and encoders."""
        if base_dir is None:
            self.base_dir = Path(__file__).resolve().parent.parent
        else:
            self.base_dir = Path(base_dir)

        model_path = self.base_dir / "models" / "fertilizer_model.pkl"
        scaler_path = self.base_dir / "scalers" / "fertilizer_scaler.pkl"
        encoder_path = self.base_dir / "encoders" / "fertilizer_encoders.pkl"

        if not model_path.exists() or not scaler_path.exists() or not encoder_path.exists():
            raise FileNotFoundError(
                f"Missing fertilizer artifacts in {self.base_dir}. Ensure model, scaler, and encoder exist."
            )

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.encoders = joblib.load(encoder_path)

        self.soil_encoder = self.encoders["soil_encoder"]
        self.crop_encoder = self.encoders["crop_encoder"]
        self.target_encoder = self.encoders["target_encoder"]
        self.num_cols = self.encoders["num_cols"]

        self.valid_soils = {s.lower(): s for s in self.encoders["soil_classes"]}
        self.valid_crops = {c.lower(): c for c in self.encoders["crop_classes"]}

    def normalize_soil(self, soil: str) -> str:
        """Map user-provided soil string to trained canonical soil type."""
        clean = str(soil).strip().lower()
        return self.valid_soils.get(clean, "Loamy")

    def normalize_crop(self, crop: str) -> str:
        """Map user-provided crop string to trained canonical crop type."""
        clean = str(crop).strip().lower()
        if clean in self.valid_crops:
            return self.valid_crops[clean]
        # Agronomic category fallback mapping
        if any(term in clean for term in ["paddy", "grain", "cereal"]):
            return "Rice"
        if any(term in clean for term in ["corn"]):
            return "Maize"
        if any(term in clean for term in ["pulse", "bean", "dal", "gram", "lentil"]):
            return "Chickpea"
        if any(term in clean for term in ["peanut"]):
            return "Ground Nuts"
        return "Rice"

    def analyze_nutrients(self, crop: str, n: float, p: float, k: float) -> NutrientDeficitReport:
        """Evaluate nutrient status against crop requirement benchmarks."""
        bench = self.CROP_NUTRIENT_BENCHMARKS.get(crop.lower(), self.CROP_NUTRIENT_BENCHMARKS["default"])
        
        n_ratio = n / bench["N"]
        p_ratio = p / bench["P"]
        k_ratio = k / bench["K"]

        def get_status(ratio: float) -> str:
            if ratio < 0.75:
                return "Deficient"
            if ratio > 1.30:
                return "Surplus"
            return "Optimal"

        n_stat = get_status(n_ratio)
        p_stat = get_status(p_ratio)
        k_stat = get_status(k_ratio)

        # Identify most limiting nutrient (lowest ratio)
        ratios = {"Nitrogen (N)": n_ratio, "Phosphorus (P)": p_ratio, "Potassium (K)": k_ratio}
        limiting = min(ratios, key=ratios.get)  # type: ignore

        return NutrientDeficitReport(
            nitrogen_status=n_stat,
            phosphorus_status=p_stat,
            potassium_status=k_stat,
            limiting_nutrient=limiting
        )

    def calculate_dosage_and_schedule(
        self,
        fertilizer_name: str,
        crop: str,
        nutrient_report: NutrientDeficitReport
    ) -> tuple[float, str, str]:
        """Compute recommended dosage per acre, application timing, and sustainable tip."""
        bench = self.CROP_NUTRIENT_BENCHMARKS.get(crop.lower(), self.CROP_NUTRIENT_BENCHMARKS["default"])
        base_dose = bench["default_dose"]

        # Modulate dosage based on limiting nutrient severity
        if nutrient_report.limiting_nutrient == "Nitrogen (N)" and fertilizer_name == "Urea":
            dosage = base_dose * 1.15
            schedule = (
                "Split Application: Apply 50% basal dose during land preparation/sowing, "
                "25% at tillering/vegetative growth (30 DAS), and 25% at panicle/flowering initiation."
            )
            sustainability = (
                "Eco-Tip: Use Neem-Coated Urea to reduce nitrate leaching and greenhouse gas emissions "
                "by 15-20%. Avoid applying right before heavy rainfall."
            )
        elif nutrient_report.limiting_nutrient == "Phosphorus (P)" and fertilizer_name in ["DAP", "14-35-14"]:
            dosage = base_dose * 1.10
            schedule = (
                "Basal Placement: Apply 100% of DAP/phosphatic fertilizer at the time of sowing "
                "placed 5 cm below and to the side of the seed line for optimal root development."
            )
            sustainability = (
                "Eco-Tip: Integrate Phosphorus Solubilizing Bacteria (PSB bio-fertilizer) "
                "to enhance uptake efficiency and prevent phosphorus fixation in the soil."
            )
        elif fertilizer_name == "10-26-26":
            dosage = base_dose * 1.05
            schedule = (
                "Split Application: 75% as basal dose at sowing, 25% as top dressing during "
                "early vegetative growth stage."
            )
            sustainability = (
                "Eco-Tip: Incorporate organic compost or green manure to boost soil cation exchange capacity (CEC) "
                "and optimize potash retention."
            )
        else:
            dosage = base_dose
            schedule = "Standard Practice: Apply 50% at sowing as basal dose, remainder top-dressed at 30-40 DAS."
            sustainability = (
                "Eco-Tip: Regularly test soil organic carbon (SOC); maintain organic mulch to protect microbial diversity."
            )

        return round(dosage, 1), schedule, sustainability

    def predict(self, input_data: Union[FertilizerInputSchema, Dict[str, Any]]) -> FertilizerPredictionResponse:
        """
        Predict optimal fertilizer and generate actionable agronomic instructions.
        """
        if isinstance(input_data, dict):
            validated = FertilizerInputSchema(**input_data)
        else:
            validated = input_data

        canonical_soil = self.normalize_soil(validated.soil_type)
        canonical_crop = self.normalize_crop(validated.crop_type)

        soil_idx = self.soil_encoder.transform([canonical_soil])[0]
        crop_idx = self.crop_encoder.transform([canonical_crop])[0]

        num_df = pd.DataFrame(
            [[
                validated.temperature,
                validated.humidity,
                validated.moisture,
                validated.nitrogen,
                validated.potassium,
                validated.phosphorus,
            ]],
            columns=self.num_cols
        )
        scaled_num = self.scaler.transform(num_df)
        feature_matrix = np.hstack([scaled_num, [[soil_idx, crop_idx]]])

        pred_idx = self.model.predict(feature_matrix)[0]
        fertilizer_name = str(self.target_encoder.inverse_transform([pred_idx])[0])

        probabilities = self.model.predict_proba(feature_matrix)[0]
        confidence = float(np.max(probabilities))

        nutrient_report = self.analyze_nutrients(
            crop=canonical_crop,
            n=validated.nitrogen,
            p=validated.phosphorus,
            k=validated.potassium
        )

        dosage, schedule, sustainability_tip = self.calculate_dosage_and_schedule(
            fertilizer_name=fertilizer_name,
            crop=canonical_crop,
            nutrient_report=nutrient_report
        )

        return FertilizerPredictionResponse(
            recommended_fertilizer=fertilizer_name,
            confidence=round(confidence, 4),
            dosage_kg_per_acre=dosage,
            application_schedule=schedule,
            nutrient_report=nutrient_report,
            sustainability_tip=sustainability_tip
        )


# Global singleton instance
_fertilizer_service_instance = None


def get_fertilizer_service() -> FertilizerRecommendationService:
    """Get or initialize singleton instance of FertilizerRecommendationService."""
    global _fertilizer_service_instance
    if _fertilizer_service_instance is None:
        _fertilizer_service_instance = FertilizerRecommendationService()
    return _fertilizer_service_instance


def recommend_fertilizer(
    nitrogen: float,
    phosphorus: float,
    potassium: float,
    temperature: float = 26.0,
    humidity: float = 65.0,
    moisture: float = 40.0,
    soil_type: str = "Loamy",
    crop_type: str = "Rice"
) -> Dict[str, Any]:
    """Convenience helper to invoke fertilizer prediction with individual parameters."""
    service = get_fertilizer_service()
    payload = FertilizerInputSchema(
        nitrogen=nitrogen,
        phosphorus=phosphorus,
        potassium=potassium,
        temperature=temperature,
        humidity=humidity,
        moisture=moisture,
        soil_type=soil_type,
        crop_type=crop_type
    )
    return service.predict(payload).model_dump()
