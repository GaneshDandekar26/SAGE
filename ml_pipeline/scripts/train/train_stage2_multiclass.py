# ml_pipeline/scripts/train/train_stage2_multiclass.py
"""
Train Stage 2: Flood vs Scraper vs Recon (3-class Random Forest)

This model only receives traffic already confirmed as "Bot" by Stage 1.
Its job is to classify the *type* of attack so the gateway can apply
the correct graduated response (BAN / RATE_LIMIT / CAPTCHA).

Model:  RandomForestClassifier
Input:  10 operational features (from generate_stage2_features.py)
Output: sage_stage2_bot_classifier.pkl + sage_stage2_label_encoder.pkl
"""

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import LabelEncoder


def resolve_base_dir():
    current = Path(__file__).resolve().parent
    for candidate in [current, *current.parents]:
        if candidate.name == "ml_pipeline" and (candidate / "requirements.txt").exists():
            return str(candidate)
    raise RuntimeError("Could not resolve ml_pipeline base directory.")


BASE_DIR = resolve_base_dir()

STAGE2_FEATURES = [
    "SAGE_Request_Velocity",
    "SAGE_Peak_Burst_RPS",
    "SAGE_Velocity_Trend",
    "SAGE_Endpoint_Concentration",
    "SAGE_Cart_Ratio",
    "SAGE_Sequential_Traversal",
    "SAGE_Sensitive_Endpoint_Ratio",
    "SAGE_UA_Entropy",
    "SAGE_Header_Completeness",
    "SAGE_Response_Size_Variance",
]


def main():
    parser = argparse.ArgumentParser(description="Train Stage 2: Bot classification")
    parser.add_argument(
        "--input",
        default=os.path.join(BASE_DIR, "data", "stage2_training_data.csv"),
        help="Stage 2 training CSV",
    )
    parser.add_argument(
        "--output-model",
        default=os.path.join(BASE_DIR, "models", "sage_stage2_bot_classifier.pkl"),
        help="Output model path",
    )
    parser.add_argument(
        "--output-encoder",
        default=os.path.join(BASE_DIR, "models", "sage_stage2_label_encoder.pkl"),
        help="Output label encoder path",
    )
    parser.add_argument(
        "--output-report",
        default=os.path.join(BASE_DIR, "reports", "stage2_classification_report.json"),
        help="Output report path",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("SAGE Stage 2: Flood vs Scraper vs Recon — Random Forest")
    print("=" * 60)

    # 1. Load data
    print(f"\n[1] Loading {args.input}...")
    df = pd.read_csv(args.input)
    df = df.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    X = df[STAGE2_FEATURES]
    y_raw = df["label"]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    print(f"    Total rows: {len(df)}")
    print(f"    Classes: {label_encoder.classes_.tolist()}")
    print(f"    Class distribution:\n{y_raw.value_counts().to_string()}")

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Train
    print("\n[2] Training RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_leaf=5,
        max_samples=0.8,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # 4. Cross-validation
    print("\n[3] Running 5-fold cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(
        model, X, y, cv=cv,
        scoring=["accuracy", "f1_macro", "recall_macro", "precision_macro"],
        n_jobs=-1,
    )

    for metric in ["accuracy", "f1_macro", "recall_macro", "precision_macro"]:
        scores = cv_results[f"test_{metric}"]
        print(f"    CV {metric}: {scores.mean():.4f} (±{scores.std():.4f})")

    # 5. Holdout evaluation
    print("\n[4] Evaluating on holdout set...")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    class_names = label_encoder.classes_.tolist()
    report = classification_report(
        y_test, y_pred,
        target_names=class_names,
        output_dict=True,
    )

    print(f"    Accuracy:   {accuracy:.4f}")
    print(f"    Macro F1:   {report['macro avg']['f1-score']:.4f}")
    print(f"    Macro Recall: {report['macro avg']['recall']:.4f}")

    for cls in class_names:
        print(f"    {cls} — P: {report[cls]['precision']:.4f}  "
              f"R: {report[cls]['recall']:.4f}  "
              f"F1: {report[cls]['f1-score']:.4f}")

    print(f"\n    Confusion Matrix:")
    print(f"    {cm}")

    # 6. Feature importance
    importance_data = sorted(
        zip(STAGE2_FEATURES, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )
    print("\n    Feature Importances:")
    for name, score in importance_data:
        print(f"      {name}: {score:.4f}")

    # 7. Save
    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_report), exist_ok=True)

    joblib.dump(model, args.output_model, compress=3)
    joblib.dump(label_encoder, args.output_encoder, compress=3)

    report_payload = {
        "stage": "stage2_bot_classification",
        "algorithm": "RandomForestClassifier",
        "classes": class_names,
        "features": STAGE2_FEATURES,
        "dataset_rows": len(df),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "class_distribution": {k: int(v) for k, v in y_raw.value_counts().to_dict().items()},
        "cross_validation": {
            "mean_accuracy": float(cv_results["test_accuracy"].mean()),
            "mean_f1_macro": float(cv_results["test_f1_macro"].mean()),
            "mean_recall_macro": float(cv_results["test_recall_macro"].mean()),
        },
        "holdout_metrics": {
            "accuracy": float(accuracy),
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
        },
        "feature_importance": [
            {"feature": n, "importance": float(s)} for n, s in importance_data
        ],
    }

    with open(args.output_report, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Model saved:   {args.output_model}")
    print(f"Encoder saved: {args.output_encoder}")
    print(f"Report saved:  {args.output_report}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
