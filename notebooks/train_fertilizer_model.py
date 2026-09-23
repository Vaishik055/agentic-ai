"""
Fertilizer Recommendation Model Training Pipeline.
Generates balanced agronomic dataset, trains a Random Forest Classifier,
evaluates cross-validated performance, and serializes artifacts for production inference.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


def generate_and_train():
    """Generate training data and train the Fertilizer Recommendation model."""
    root_dir = Path(__file__).resolve().parent.parent
    data_dir = root_dir / "datasets"
    model_dir = root_dir / "backend" / "models"
    scaler_dir = root_dir / "backend" / "scalers"
    encoder_dir = root_dir / "backend" / "encoders"

    data_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    scaler_dir.mkdir(parents=True, exist_ok=True)
    encoder_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)

    soil_types = ["Sandy", "Loamy", "Black", "Red", "Clayey"]
    crop_types = ["Rice", "Maize", "Chickpea", "Kidneybeans", "Pigeonpeas", "Cotton", "Sugarcane", "Wheat", "Ground Nuts"]
    fertilizers = ["Urea", "DAP", "14-35-14", "28-28", "17-17-17", "20-20", "10-26-26"]

    records = []
    for _ in range(2100):
        fert = np.random.choice(fertilizers)
        soil = np.random.choice(soil_types)
        crop = np.random.choice(crop_types)
        temp = np.random.uniform(20.0, 38.0)
        humidity = np.random.uniform(40.0, 85.0)
        moisture = np.random.uniform(25.0, 65.0)

        if fert == "Urea":
            n = np.random.uniform(5.0, 30.0)
            p = np.random.uniform(40.0, 80.0)
            k = np.random.uniform(40.0, 80.0)
        elif fert == "DAP":
            n = np.random.uniform(40.0, 80.0)
            p = np.random.uniform(5.0, 22.0)
            k = np.random.uniform(40.0, 80.0)
        elif fert == "10-26-26":
            n = np.random.uniform(40.0, 85.0)
            p = np.random.uniform(5.0, 25.0)
            k = np.random.uniform(5.0, 25.0)
        elif fert == "14-35-14":
            n = np.random.uniform(30.0, 60.0)
            p = np.random.uniform(8.0, 25.0)
            k = np.random.uniform(25.0, 45.0)
        elif fert == "28-28":
            n = np.random.uniform(10.0, 30.0)
            p = np.random.uniform(10.0, 25.0)
            k = np.random.uniform(40.0, 80.0)
        elif fert == "17-17-17":
            n = np.random.uniform(35.0, 55.0)
            p = np.random.uniform(35.0, 55.0)
            k = np.random.uniform(35.0, 55.0)
        else:  # 20-20
            n = np.random.uniform(20.0, 40.0)
            p = np.random.uniform(20.0, 40.0)
            k = np.random.uniform(45.0, 80.0)

        records.append([
            round(temp, 1),
            round(humidity, 1),
            round(moisture, 1),
            soil,
            crop,
            round(n, 1),
            round(k, 1),
            round(p, 1),
            fert
        ])

    df = pd.DataFrame(
        records,
        columns=["temperature", "humidity", "moisture", "soil_type", "crop_type", "nitrogen", "potassium", "phosphorus", "fertilizer"]
    )
    dataset_path = data_dir / "fertilizer_prediction.csv"
    df.to_csv(dataset_path, index=False)

    le_soil = LabelEncoder()
    le_crop = LabelEncoder()
    le_fert = LabelEncoder()

    df["soil_encoded"] = le_soil.fit_transform(df["soil_type"])
    df["crop_encoded"] = le_crop.fit_transform(df["crop_type"])
    df["target"] = le_fert.fit_transform(df["fertilizer"])

    num_cols = ["temperature", "humidity", "moisture", "nitrogen", "potassium", "phosphorus"]
    scaler = StandardScaler()
    scaled_num = scaler.fit_transform(df[num_cols])

    feature_matrix = np.hstack([scaled_num, df[["soil_encoded", "crop_encoded"]].values])
    X_train, X_test, y_train, y_test = train_test_split(
        feature_matrix,
        df["target"].values,
        test_size=0.2,
        random_state=42,
        stratify=df["target"].values
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"Training Complete. Train Accuracy: {train_score*100:.2f}%, Test Accuracy: {test_score*100:.2f}%")

    encoders = {
        "soil_encoder": le_soil,
        "crop_encoder": le_crop,
        "target_encoder": le_fert,
        "num_cols": num_cols,
        "soil_classes": [str(c).title() for c in le_soil.classes_],
        "crop_classes": [str(c).title() for c in le_crop.classes_],
        "fertilizer_classes": list(le_fert.classes_)
    }

    joblib.dump(model, model_dir / "fertilizer_model.pkl")
    joblib.dump(scaler, scaler_dir / "fertilizer_scaler.pkl")
    joblib.dump(encoders, encoder_dir / "fertilizer_encoders.pkl")
    print("Fertilizer artifacts successfully saved to backend directories.")


if __name__ == "__main__":
    generate_and_train()
