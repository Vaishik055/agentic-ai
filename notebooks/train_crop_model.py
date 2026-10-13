# ==============================================================================
# Crop Recommendation Model Training Pipeline
# Trains a high-precision Random Forest Classifier on the 22-crop dataset,
# evaluates cross-validation performance, and serializes production artifacts.
# ==============================================================================

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score


def train_crop_recommendation_model():
    """Trains and serializes the 22-Crop Random Forest Model."""
    root_dir = Path(__file__).resolve().parent.parent
    data_dir = root_dir / "datasets"
    model_dir = root_dir / "backend" / "models"
    scaler_dir = root_dir / "backend" / "scalers"
    encoder_dir = root_dir / "backend" / "encoders"

    model_dir.mkdir(parents=True, exist_ok=True)
    scaler_dir.mkdir(parents=True, exist_ok=True)
    encoder_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = data_dir / "crop_recommendation.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing dataset at {dataset_path}")

    print("=" * 70)
    print("🌾 Training Crop Recommendation Model (22 Crops, 7 Features)")
    print("=" * 70)

    df = pd.read_csv(dataset_path)
    feature_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    X = df[feature_cols]
    y = df["label"]

    print(f"📊 Dataset Loaded: {df.shape[0]} samples across {df['label'].nunique()} crop classes.")

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train / Test split (80/20 stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    # Train Random Forest Classifier
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    cv_scores = cross_val_score(model, X_scaled, y_encoded, cv=5)

    print(f"🎯 Train Accuracy:              {train_acc * 100:.2f}%")
    print(f"🎯 Test Accuracy (80/20 split):  {test_acc * 100:.2f}%")
    print(f"🔄 5-Fold Stratified CV Acc:    {cv_scores.mean() * 100:.2f}% (± {cv_scores.std() * 100:.2f}%)")

    # Serialize artifacts
    joblib.dump(model, model_dir / "crop_model.pkl")
    joblib.dump(scaler, scaler_dir / "crop_scaler.pkl")
    joblib.dump(label_encoder, encoder_dir / "label_encoder.pkl")

    print("\n✅ Model artifacts successfully saved:")
    print(f"  - Model:   {model_dir / 'crop_model.pkl'}")
    print(f"  - Scaler:  {scaler_dir / 'crop_scaler.pkl'}")
    print(f"  - Encoder: {encoder_dir / 'label_encoder.pkl'}")


if __name__ == "__main__":
    train_crop_recommendation_model()