import argparse
import json
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


FEATURES = [
    "SAGE_Session_Depth",
    "SAGE_Temporal_Variance",
    "SAGE_Request_Velocity",
    "SAGE_Behavioral_Diversity",
    "SAGE_Endpoint_Concentration",
    "SAGE_Cart_Ratio",
    "SAGE_Asset_Skip_Ratio",
    "SAGE_Sequential_Traversal",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Retrain SAGE model from labeled gateway telemetry CSV.")
    parser.add_argument("--input", default="ml_pipeline/data/training_data_balanced.csv", help="Input CSV file")
    parser.add_argument("--output-model", default="ml_pipeline/models/sage_model.pkl", help="Output model file")
    parser.add_argument("--output-report", default="ml_pipeline/reports/classification_report.json", help="Output report JSON")
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.input)
    df = df[df["label"] != "unknown"].copy()

    if df.empty:
        raise RuntimeError("No labeled rows found. Check Kafka labeling/export first.")

    missing = [c for c in FEATURES + ["label"] if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    X = df[FEATURES]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred))

    importance = sorted(
        zip(FEATURES, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )

    print("\nFeature importances:")
    for name, score in importance:
        print(f"{name}: {score:.4f}")

    joblib.dump(model, args.output_model)

    report_payload = {
        "input": os.path.abspath(args.input),
        "model": os.path.abspath(args.output_model),
        "rows_total": int(len(df)),
        "class_counts": {k: int(v) for k, v in df["label"].value_counts().to_dict().items()},
        "classification_report": report_dict,
        "feature_importance": [{"feature": n, "importance": float(s)} for n, s in importance],
    }
    with open(args.output_report, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    print(f"\nSaved model: {args.output_model}")
    print(f"Saved report: {args.output_report}")


if __name__ == "__main__":
    main()
