import argparse
from pathlib import Path

import joblib
import pandas as pd


FEATURE_COLUMNS = [
    "mfcc_1",
    "mfcc_2",
    "mfcc_3",
    "mfcc_4",
    "mfcc_5",
    "mfcc_6",
    "mfcc_7",
    "mfcc_8",
    "mfcc_9",
    "mfcc_10",
    "mfcc_11",
    "mfcc_12",
    "mfcc_13",
    "spectral_centroid",
    "spectral_rolloff",
    "zcr",
    "rms",
    "peak_frequency",
    "mean",
    "std",
]


def load_model(repo_root: Path):
    model_path = repo_root / "ml" / "heart_random_forest_model_jkall.pkl"
    encoder_path = repo_root / "ml" / "label_encoder_jkall.pkl"

    model = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)

    return model, label_encoder


def predict_from_features(model, label_encoder, features: pd.DataFrame):
    features = features[FEATURE_COLUMNS]

    prediction_number = model.predict(features)[0]
    prediction_label = label_encoder.inverse_transform([prediction_number])[0]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        confidence = max(probabilities)
    else:
        confidence = None

    return prediction_label, confidence


def main():
    parser = argparse.ArgumentParser(
        description="Run Raspberry Pi inference for the smart stethoscope heart model."
    )

    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional CSV file containing extracted heart sound features.",
    )

    parser.add_argument(
        "--row",
        type=int,
        default=0,
        help="CSV row number to test.",
    )

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    model, label_encoder = load_model(repo_root)

    if args.csv is None:
        csv_path = repo_root / "ml" / "data" / "heart_features_jkall.csv"
    else:
        csv_path = Path(args.csv)

    df = pd.read_csv(csv_path)

    if args.row < 0 or args.row >= len(df):
        raise ValueError(f"Row {args.row} is out of range. CSV has {len(df)} rows.")

    sample = df.iloc[[args.row]]

    prediction_label, confidence = predict_from_features(
        model,
        label_encoder,
        sample,
    )

    print("====================================")
    print("Smart Stethoscope Raspberry Pi Inference")
    print("====================================")
    print(f"CSV file: {csv_path}")
    print(f"Row used: {args.row}")
    print(f"Prediction: {prediction_label}")

    if confidence is not None:
        print(f"Confidence: {confidence:.2%}")


if __name__ == "__main__":
    main()