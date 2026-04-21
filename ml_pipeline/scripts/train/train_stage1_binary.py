"""
SAGE Stage 1 — Human vs Bot (Binary XGBoost Classifier)
========================================================

Purpose:
    Gatekeeper model. Its #1 priority is to NEVER block real human traffic.
    Uses 12 behavioural features that capture timing micro-patterns,
    navigation habits, and content-interaction signals.

Training strategy:
    - XGBClassifier with scale_pos_weight for class weighting (no SMOTE)
    - Threshold tuned for ≥ 90% Human recall
    - 80/20 stratified train/test split
    - 5-fold stratified cross-validation
    - Reproducible (random_state=42)

Outputs:
    - models/human_vs_bot.pkl         — trained model
    - models/human_vs_bot_threshold.json — optimal decision threshold
    - reports/human_vs_bot.txt        — human-readable evaluation report
"""

import argparse
import json
import os
import sys
import textwrap
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False
    from sklearn.ensemble import GradientBoostingClassifier


# ── Constants ────────────────────────────────────────────────────────

RANDOM_STATE = 42

STAGE1_FEATURES = [
    "SAGE_InterArrival_CV",
    "SAGE_Timing_Entropy",
    "SAGE_Pause_Ratio",
    "SAGE_Burst_Score",
    "SAGE_Backtrack_Ratio",
    "SAGE_Path_Entropy",
    "SAGE_Referral_Chain_Depth",
    "SAGE_Session_Depth",
    "SAGE_Method_Diversity",
    "SAGE_Static_Asset_Ratio",
    "SAGE_Error_Rate",
    "SAGE_Payload_Variance",
]


def _resolve_base_dir() -> str:
    """Walk up from this file until we find the ml_pipeline root."""
    current = Path(__file__).resolve().parent
    for candidate in [current, *current.parents]:
        if candidate.name == "ml_pipeline" and (candidate / "requirements.txt").exists():
            return str(candidate)
    raise RuntimeError("Could not resolve ml_pipeline base directory.")


BASE_DIR = _resolve_base_dir()


# ── Threshold Search ─────────────────────────────────────────────────

def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    min_human_recall: float = 0.90,
) -> float:
    """
    Sweep thresholds 0.05–0.95 and return the one that maximises overall
    accuracy while keeping Human recall (class 0) ≥ min_human_recall.
    """
    best_threshold = 0.50
    best_accuracy = 0.0

    for t in np.arange(0.05, 0.96, 0.01):
        preds = (y_proba >= t).astype(int)

        # Human recall = (true humans predicted as 0) / (all true humans)
        human_mask = y_true == 0
        if human_mask.sum() == 0:
            continue
        human_recall = (preds[human_mask] == 0).sum() / human_mask.sum()

        if human_recall < min_human_recall:
            continue

        acc = accuracy_score(y_true, preds)
        if acc > best_accuracy:
            best_accuracy = acc
            best_threshold = float(round(t, 2))

    return best_threshold


# ── Report Writer ────────────────────────────────────────────────────

def write_text_report(
    path: str,
    *,
    algorithm: str,
    features: list[str],
    dataset_rows: int,
    train_rows: int,
    test_rows: int,
    class_dist: dict,
    cv_results: dict,
    accuracy: float,
    roc_auc: float,
    report_str: str,
    cm: np.ndarray,
    threshold: float,
    importance: list[tuple[str, float]],
) -> None:
    """Write a human-readable evaluation report as plain text."""

    tn, fp, fn, tp = cm.ravel()

    lines = [
        "=" * 70,
        "  SAGE Stage 1 — Human vs Bot Evaluation Report",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
        "MODEL",
        f"  Algorithm:          {algorithm}",
        f"  Features:           {len(features)}",
        f"  Optimal Threshold:  {threshold}",
        "",
        "DATASET",
        f"  Total rows:         {dataset_rows:,}",
        f"  Train rows:         {train_rows:,}",
        f"  Test rows:          {test_rows:,}",
        f"  Human (label=0):    {class_dist.get(0, 0):,}",
        f"  Bot   (label=1):    {class_dist.get(1, 0):,}",
        "",
        "CROSS-VALIDATION (5-fold stratified)",
    ]
    for metric in ["accuracy", "f1", "recall", "precision", "roc_auc"]:
        key = f"test_{metric}"
        if key in cv_results:
            vals = cv_results[key]
            lines.append(f"  {metric:<12s}  {vals.mean():.4f}  (±{vals.std():.4f})")

    lines += [
        "",
        "HOLDOUT METRICS",
        f"  Accuracy:   {accuracy:.4f}",
        f"  ROC-AUC:    {roc_auc:.4f}",
        "",
        "CONFUSION MATRIX",
        f"  True Negatives  (humans correctly allowed):   {tn:,}",
        f"  False Positives (humans incorrectly blocked): {fp:,}",
        f"  False Negatives (bots missed):                {fn:,}",
        f"  True Positives  (bots correctly blocked):     {tp:,}",
        "",
        "CLASSIFICATION REPORT",
        report_str,
        "",
        "FEATURE IMPORTANCE",
    ]
    for name, score in importance:
        bar = "█" * int(score * 50)
        lines.append(f"  {name:<30s}  {score:.4f}  {bar}")

    lines += [
        "",
        "FEATURES USED",
    ]
    for i, f in enumerate(features, 1):
        lines.append(f"  {i:2d}. {f}")

    lines.append("")
    lines.append("=" * 70)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train Stage 1: Human vs Bot binary classifier"
    )
    parser.add_argument(
        "--input",
        default=os.path.join(BASE_DIR, "data", "stage1_training_data.csv"),
    )
    parser.add_argument(
        "--output-model",
        default=os.path.join(BASE_DIR, "models", "human_vs_bot.pkl"),
    )
    parser.add_argument(
        "--output-report",
        default=os.path.join(BASE_DIR, "reports", "human_vs_bot.txt"),
    )
    args = parser.parse_args()

    print("=" * 60)
    print("SAGE Stage 1: Human vs Bot — Binary Classifier")
    print("=" * 60)

    # ── 1. Load ──────────────────────────────────────────────────────
    print(f"\n[1/6] Loading {args.input} ...")
    df = pd.read_csv(args.input)
    df = df.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    missing = [c for c in STAGE1_FEATURES + ["label"] if c not in df.columns]
    if missing:
        print(f"FATAL: Missing columns: {missing}", file=sys.stderr)
        sys.exit(1)

    X = df[STAGE1_FEATURES]
    y = df["label"]

    n_human = int((y == 0).sum())
    n_bot = int((y == 1).sum())
    print(f"    Rows: {len(df):,}  |  Human: {n_human:,}  |  Bot: {n_bot:,}")

    # ── 2. Split ─────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y,
    )

    # ── 3. Train ─────────────────────────────────────────────────────
    print("\n[2/6] Training model ...")

    # Use class weights via scale_pos_weight (no SMOTE)
    pos_weight = n_human / max(n_bot, 1)

    if _HAS_XGB:
        algo_name = "XGBClassifier"
        model = XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=pos_weight,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            n_jobs=-1,
            verbosity=0,
        )
    else:
        algo_name = "GradientBoostingClassifier"
        print("    [WARN] xgboost not installed — using sklearn fallback")
        model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            random_state=RANDOM_STATE,
        )

    model.fit(X_train, y_train)

    # ── 4. Cross-validation ──────────────────────────────────────────
    print("\n[3/6] 5-fold cross-validation ...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_results = cross_validate(
        model, X, y, cv=cv,
        scoring=["accuracy", "f1", "recall", "precision", "roc_auc"],
        n_jobs=-1,
    )
    for metric in ["accuracy", "f1", "recall", "precision", "roc_auc"]:
        vals = cv_results[f"test_{metric}"]
        print(f"    {metric:<12s} {vals.mean():.4f} (±{vals.std():.4f})")

    # ── 5. Holdout evaluation ────────────────────────────────────────
    print("\n[4/6] Holdout evaluation ...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    report_str = classification_report(
        y_test, y_pred, target_names=["Human", "Bot"]
    )

    print(f"    Accuracy:     {accuracy:.4f}")
    print(f"    ROC-AUC:      {roc_auc:.4f}")
    print(f"    Human Recall: {1 - fp / max(tn + fp, 1):.4f}")
    print(f"    Bot Recall:   {tp / max(tp + fn, 1):.4f}")
    print(f"    FP (humans blocked): {fp}")
    print(f"    FN (bots missed):    {fn}")

    # ── 6. Optimal threshold ─────────────────────────────────────────
    print("\n[5/6] Finding optimal threshold (Human recall ≥ 90%) ...")
    threshold = find_optimal_threshold(y_test.values, y_proba, min_human_recall=0.90)
    print(f"    Threshold: {threshold}")

    # ── 7. Feature importance ────────────────────────────────────────
    importance = sorted(
        zip(STAGE1_FEATURES, model.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )
    print("\n    Feature Importances:")
    for name, score in importance:
        print(f"      {name:<30s} {score:.4f}")

    # ── 8. Save ──────────────────────────────────────────────────────
    print(f"\n[6/6] Saving artifacts ...")
    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    joblib.dump(model, args.output_model, compress=3)

    threshold_path = os.path.join(
        os.path.dirname(args.output_model), "human_vs_bot_threshold.json"
    )
    with open(threshold_path, "w") as f:
        json.dump({"optimal_threshold": threshold}, f, indent=2)

    write_text_report(
        args.output_report,
        algorithm=algo_name,
        features=STAGE1_FEATURES,
        dataset_rows=len(df),
        train_rows=len(X_train),
        test_rows=len(X_test),
        class_dist={0: n_human, 1: n_bot},
        cv_results=cv_results,
        accuracy=accuracy,
        roc_auc=roc_auc,
        report_str=report_str,
        cm=cm,
        threshold=threshold,
        importance=importance,
    )

    print(f"    Model:     {args.output_model}")
    print(f"    Threshold: {threshold_path}")
    print(f"    Report:    {args.output_report}")
    print("=" * 60)


if __name__ == "__main__":
    main()
