"""
SAGE Stage 2 — Attack Classifier (Flood / Scraper / Recon)
============================================================

Purpose:
    Classify confirmed bots into one of three attack categories so the
    gateway can apply the correct graduated response:
        Flood   → IP BAN
        Scraper → RATE LIMIT
        Recon   → CAPTCHA

Training strategy:
    - Trained ONLY on Bot data (no Human rows)
    - RandomForestClassifier with class_weight='balanced' (no SMOTE)
    - 80/20 stratified train/test split
    - 5-fold stratified cross-validation
    - Reproducible (random_state=42)

Outputs:
    - models/attack_classifier.pkl        — trained model
    - models/attack_classifier_encoder.pkl — label encoder
    - reports/attack_classifier.txt        — human-readable report
"""

import argparse
import json
import os
import sys
from datetime import datetime
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


# ── Constants ────────────────────────────────────────────────────────

RANDOM_STATE = 42

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


def _resolve_base_dir() -> str:
    current = Path(__file__).resolve().parent
    for candidate in [current, *current.parents]:
        if candidate.name == "ml_pipeline" and (candidate / "requirements.txt").exists():
            return str(candidate)
    raise RuntimeError("Could not resolve ml_pipeline base directory.")


BASE_DIR = _resolve_base_dir()


# ── Report Writer ────────────────────────────────────────────────────

def write_text_report(
    path: str,
    *,
    features: list[str],
    classes: list[str],
    dataset_rows: int,
    train_rows: int,
    test_rows: int,
    class_dist: dict,
    cv_results: dict,
    accuracy: float,
    report_str: str,
    cm: np.ndarray,
    importance: list[tuple[str, float]],
) -> None:
    """Write a human-readable evaluation report as plain text."""
    lines = [
        "=" * 70,
        "  SAGE Stage 2 — Attack Classifier Evaluation Report",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
        "MODEL",
        "  Algorithm:    RandomForestClassifier",
        f"  Features:     {len(features)}",
        f"  Classes:      {', '.join(classes)}",
        "",
        "DATASET (Bot-only — no human rows)",
        f"  Total rows:   {dataset_rows:,}",
        f"  Train rows:   {train_rows:,}",
        f"  Test rows:    {test_rows:,}",
    ]
    for cls, count in sorted(class_dist.items()):
        lines.append(f"  {cls:<12s}   {count:,}")

    lines += [
        "",
        "CROSS-VALIDATION (5-fold stratified)",
    ]
    for metric in ["accuracy", "f1_macro", "recall_macro", "precision_macro"]:
        key = f"test_{metric}"
        if key in cv_results:
            vals = cv_results[key]
            lines.append(f"  {metric:<18s}  {vals.mean():.4f}  (±{vals.std():.4f})")

    lines += [
        "",
        "HOLDOUT METRICS",
        f"  Accuracy:      {accuracy:.4f}",
        "",
        "CONFUSION MATRIX",
    ]

    # Format confusion matrix as a labeled table
    col_header = "          " + "  ".join(f"{c:>10s}" for c in classes)
    lines.append(col_header)
    for i, row_label in enumerate(classes):
        row_vals = "  ".join(f"{cm[i, j]:>10d}" for j in range(len(classes)))
        lines.append(f"  {row_label:<8s} {row_vals}")

    lines += [
        "",
        "CLASSIFICATION REPORT",
        report_str,
        "",
        "FEATURE IMPORTANCE",
    ]
    for name, score in importance:
        bar = "█" * int(score * 50)
        lines.append(f"  {name:<35s}  {score:.4f}  {bar}")

    lines += [
        "",
        "FEATURES USED",
    ]
    for i, f in enumerate(features, 1):
        lines.append(f"  {i:2d}. {f}")

    lines += [
        "",
        "GRADUATED RESPONSE MAP",
        "  Flood   → IP BAN      (5 min)",
        "  Scraper → RATE LIMIT  (10 min)",
        "  Recon   → CAPTCHA     (30 min)",
        "",
        "=" * 70,
    ]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train Stage 2: Attack classifier (Flood/Scraper/Recon)"
    )
    parser.add_argument(
        "--input",
        default=os.path.join(BASE_DIR, "data", "stage2_training_data.csv"),
    )
    parser.add_argument(
        "--output-model",
        default=os.path.join(BASE_DIR, "models", "attack_classifier.pkl"),
    )
    parser.add_argument(
        "--output-encoder",
        default=os.path.join(BASE_DIR, "models", "attack_classifier_encoder.pkl"),
    )
    parser.add_argument(
        "--output-report",
        default=os.path.join(BASE_DIR, "reports", "attack_classifier.txt"),
    )
    args = parser.parse_args()

    print("=" * 60)
    print("SAGE Stage 2: Attack Classifier — Flood / Scraper / Recon")
    print("=" * 60)

    # ── 1. Load ──────────────────────────────────────────────────────
    print(f"\n[1/5] Loading {args.input} ...")
    df = pd.read_csv(args.input)
    df = df.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    missing = [c for c in STAGE2_FEATURES + ["label"] if c not in df.columns]
    if missing:
        print(f"FATAL: Missing columns: {missing}", file=sys.stderr)
        sys.exit(1)

    # Verify no human rows leaked in
    if "human" in df["label"].values:
        print("WARNING: Removing human rows — Stage 2 is bot-only")
        df = df[df["label"] != "human"].copy()

    X = df[STAGE2_FEATURES]
    y_raw = df["label"]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    classes = label_encoder.classes_.tolist()

    class_dist = {k: int(v) for k, v in y_raw.value_counts().to_dict().items()}
    print(f"    Rows:    {len(df):,}")
    print(f"    Classes: {classes}")
    for cls, count in sorted(class_dist.items()):
        print(f"      {cls}: {count:,}")

    # ── 2. Split ─────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y,
    )

    # ── 3. Train ─────────────────────────────────────────────────────
    print("\n[2/5] Training RandomForestClassifier ...")
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_leaf=5,
        max_samples=0.8,
        class_weight="balanced",     # handles imbalance via class weights (no SMOTE)
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # ── 4. Cross-validation ──────────────────────────────────────────
    print("\n[3/5] 5-fold cross-validation ...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_results = cross_validate(
        model, X, y, cv=cv,
        scoring=["accuracy", "f1_macro", "recall_macro", "precision_macro"],
        n_jobs=-1,
    )
    for metric in ["accuracy", "f1_macro", "recall_macro", "precision_macro"]:
        vals = cv_results[f"test_{metric}"]
        print(f"    {metric:<18s} {vals.mean():.4f} (±{vals.std():.4f})")

    # ── 5. Holdout evaluation ────────────────────────────────────────
    print("\n[4/5] Holdout evaluation ...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    report_str = classification_report(y_test, y_pred, target_names=classes)

    print(f"    Accuracy:   {accuracy:.4f}")
    for cls in classes:
        idx = label_encoder.transform([cls])[0]
        tp = cm[idx, idx]
        total = cm[idx].sum()
        recall = tp / max(total, 1)
        print(f"    {cls:>8s} recall: {recall:.4f}  ({tp}/{total})")

    # Feature importance
    importance = sorted(
        zip(STAGE2_FEATURES, model.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )
    print("\n    Feature Importances:")
    for name, score in importance:
        print(f"      {name:<35s} {score:.4f}")

    # ── 6. Save ──────────────────────────────────────────────────────
    print(f"\n[5/5] Saving artifacts ...")
    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)

    joblib.dump(model, args.output_model, compress=3)
    joblib.dump(label_encoder, args.output_encoder, compress=3)

    write_text_report(
        args.output_report,
        features=STAGE2_FEATURES,
        classes=classes,
        dataset_rows=len(df),
        train_rows=len(X_train),
        test_rows=len(X_test),
        class_dist=class_dist,
        cv_results=cv_results,
        accuracy=accuracy,
        report_str=report_str,
        cm=cm,
        importance=importance,
    )

    print(f"    Model:   {args.output_model}")
    print(f"    Encoder: {args.output_encoder}")
    print(f"    Report:  {args.output_report}")
    print("=" * 60)


if __name__ == "__main__":
    main()
