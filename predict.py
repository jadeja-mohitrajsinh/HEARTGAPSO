import argparse
import json
import pickle

import pandas as pd

from src.data_loader import FEATURE_NAMES


def main():
    parser = argparse.ArgumentParser(description="Predict heart-disease presence")
    parser.add_argument("--input", help="CSV with the 13 feature columns")
    args = parser.parse_args()
    data = pd.read_csv(args.input, na_values=["?"]) if args.input else pd.DataFrame([{
        "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
        "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
        "ca": 0, "thal": 6,
    }])
    missing = set(FEATURE_NAMES) - set(data.columns)
    if missing:
        raise ValueError(f"Missing input columns: {sorted(missing)}")
    with open("models/gapso_random_forest.pkl", "rb") as file:
        pipeline = pickle.load(file)
    probabilities = pipeline.predict_proba(data[FEATURE_NAMES])[:, 1]
    predictions = pipeline.predict(data[FEATURE_NAMES])
    for prediction, probability in zip(predictions, probabilities):
        print(json.dumps({
            "prediction": int(prediction),
            "heart_disease_probability": float(probability),
        }))


if __name__ == "__main__":
    main()