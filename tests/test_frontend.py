"""Unit and integration tests for Week 9 Streamlit frontend client and translations."""

import pytest
from frontend.api_client import FarmAPIClient
from frontend.translations import TRANSLATIONS, get_text


def test_translation_keys_completeness():
    """Verify that all keys in English dictionary exist in Hindi and Telugu."""
    en_keys = set(TRANSLATIONS["en"].keys())
    hi_keys = set(TRANSLATIONS["hi"].keys())
    te_keys = set(TRANSLATIONS["te"].keys())

    assert len(en_keys) >= 30
    # Every English key should have a corresponding Hindi and Telugu translation
    missing_in_hi = en_keys - hi_keys
    missing_in_te = en_keys - te_keys

    assert not missing_in_hi, f"Missing translations in Hindi: {missing_in_hi}"
    assert not missing_in_te, f"Missing translations in Telugu: {missing_in_te}"


def test_get_text_helper():
    """Verify get_text helper with valid and fallback cases."""
    # English
    assert "Agentic AI" in get_text("en", "app_title")
    # Hindi
    assert "एजेंटिक एआई" in get_text("hi", "app_title")
    # Telugu
    assert "ఏజెంటిక్ AI" in get_text("te", "app_title")
    # Fallback for unknown language
    assert get_text("fr", "app_title") == get_text("en", "app_title")
    # Fallback for unknown key
    assert get_text("en", "non_existent_key", default="default_val") == "default_val"


def test_api_client_initialization():
    """Verify FarmAPIClient initialization with default and custom endpoints."""
    client = FarmAPIClient()
    assert "8000" in client.base_url
    assert client.timeout > 0

    custom = FarmAPIClient(base_url="http://example.com:9000/")
    assert custom.base_url == "http://example.com:9000"


def test_api_client_crop_prediction_fallback():
    """Verify FarmAPIClient crop prediction returns valid data even if backend is offline."""
    client = FarmAPIClient(base_url="http://invalid-host-offline:9999")
    res = client.predict_crop(
        n=90.0,
        p=42.0,
        k=43.0,
        temperature=25.0,
        humidity=80.0,
        ph=6.5,
        rainfall=200.0,
    )
    assert res is not None
    assert "recommended_crop" in res
    assert res["recommended_crop"] == "rice"
    assert res["confidence"] > 0.8


def test_api_client_fertilizer_prediction_fallback():
    """Verify FarmAPIClient fertilizer recommendation returns valid data."""
    client = FarmAPIClient(base_url="http://invalid-host-offline:9999")
    res = client.recommend_fertilizer(
        soil_type="Clayey",
        crop_type="rice",
        n=30.0,
        p=50.0,
        k=50.0,
        temperature=25.0,
        humidity=80.0,
    )
    assert res is not None
    assert "recommended_fertilizer" in res
    assert (res.get("recommended_dosage_kg_per_acre") or res.get("dosage_kg_per_acre", 0)) > 0


def test_api_client_orchestrate_fallback():
    """Verify FarmAPIClient orchestrator returns complete multi-agent response."""
    client = FarmAPIClient(base_url="http://invalid-host-offline:9999")
    res = client.orchestrate(
        query="My soil has 90 nitrogen, 42 phosphorus, 43 potassium, pH 6.5, temp 25C, humidity 80%, rainfall 200mm.",
        location="Guntur",
        farm_size_acres=2.0,
    )
    assert res is not None
    assert res.get("status") == "success"
    assert "crop_recommendation" in res
    assert "sustainability" in res
    assert "profitability" in res
    assert len(res.get("execution_trace", [])) >= 3