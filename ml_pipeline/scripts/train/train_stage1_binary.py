# ml_pipeline/scripts/train/train_stage1_binary.py
"""
Train Stage 1: Human vs Bot (Binary XGBoost Classifier)

This model serves as the GATEKEEPER — its primary job is to ensure
real human traffic is never blocked (high Human recall).

Model:  XGBClassifier
Input:  12 behavioural features (from generate_stage1_features.py)
Output: sage_stage1_human_vs_bot.pkl + sage_stage1_threshold.json
"""

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    from sklearn.ensemble import GradientBoostingClassifier


def resolve_base_dir():
    current = Path(__file__).resolve().parent
    for candidate in [current, *current.parents]:
        if candidate.name == "ml_pipeline" and (candidate / "requirements.txt").exists():
            return str(candidate)
    raise RuntimeError("Could not resolve ml_pipeline base directory.")


BASE_DIR = resolve_base_dir()

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


def find_optimal_threshold(y_true, y_proba, min_human_recall=0.90):
    """
    Find the threshold that maximises F1 while keeping
    human recall (true class=0 recall) >= min_human_recall.

    Since label 0 = human, we need:
        recall_for_class_0 >= 0.90
    Which means:
        fraction of actual humans correctly predicted as human >= 0.90
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    best_threshold = 0.5
    best_f1 = 0.0

    for t in np.arange(0.3, 0.95, 0.01):
        preds = (y_proba >= t).astype(int)

        # Human recall = correctly identified humans / all actual humans
        human_mask = y_true == 0
        if human_mask.sum() == 0:
            continue
        human_recall = (preds[human_mask] == 0).sum() / human_mask.sum()

        if human_recall < min_human_recall:
            continue

        f1 = 2 * (
            (preds == y_true).sum() / len(y_true)
        )  # simplified accuracy-as-proxy
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    return best_threshold


def main():
    parser = argparse.ArgumentParser(description="Train Stage 1: Human vs Bot")
    parser.add_argument(
        "--input",
        default=os.path.join(BASE_DIR, "data", "stage1_training_data.csv"),
        help="Stage 1 training CSV",
    )
    parser.add_argument(
        "--output-model",
        default=os.path.join(BASE_DIR, "models", "sage_stage1_human_vs_bot.pkl"),
        help="Output model path",
    )
    parser.add_argument(
        "--output-report",
        default=os.path.join(BASE_DIR, "reports", "stage1_classification_report.json"),
        help="Output report path",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("SAGE Stage 1: Human vs Bot — XGBoost Binary Classifier")
    print("=" * 60)

    # 1. Load data
    print(f"\n[1] Loading {args.input}...")
    df = pd.read_csv(args.input)
    df = df.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    X = df[STAGE1_FEATURES]
    y = df["label"]

    print(f"    Total rows: {len(df)}")
    print(f"    Class distribution:\n{y.value_counts().to_string()}")

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Build model
    print("\n[2] Training XGBoost classifier...")

    if HAS_XGB:
        model = XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1.0,  # already balanced data
            random_state=42,
            eval_metric="logloss",
            use_label_encoder=False,
            n_jobs=-1,
        )
    else:
        print("    [WARN] xgboost not installed, falling back to sklearn GradientBoosting")
        model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )

    model.fit(X_train, y_train)

    # 4. Cross-validation
    print("\n[3] Running 5-fold cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(
        model, X, y, cv=cv,
        scoring=["accuracy", "f1", "recall", "precision", "roc_auc"],
        n_jobs=-1,
    )

    for metric in ["accuracy", "f1", "recall", "precision", "roc_auc"]:
        scores = cv_results[f"test_{metric}"]
        print(f"    CV {metric}: {scores.mean():.4f} (±{scores.std():.4f})")

    # 5. Holdout evaluation
    print("\n[4] Evaluating on holdout set...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        target_names=["Human", "Bot"],
        output_dict=True,
    )

    print(f"    Accuracy: {accuracy:.4f}")
    print(f"    ROC-AUC:  {roc_auc:.4f}")
    print(f"    Human Recall: {report['Human']['recall']:.4f}")
    print(f"    Bot Recall:   {report['Bot']['recall']:.4f}")
    print(f"    Human F1:     {report['Human']['f1-score']:.4f}")
    print(f"    Bot F1:       {report['Bot']['f1-score']:.4f}")

    tn, fp, fn, tp = cm.ravel()
    print(f"\n    Confusion Matrix:")
    print(f"    TN (humans allowed): {tn}")
    print(f"    FP (humans blocked): {fp}")
    print(f"    FN (bots missed):    {fn}")
    print(f"    TP (bots blocked):   {tp}")

    # 6. Find optimal threshold for human safety
    optimal_threshold = find_optimal_threshold(
        y_test.values, y_proba, min_human_recall=0.90
    )
    print(f"\n    Optimal threshold (human recall ≥ 90%): {optimal_threshold:.2f}")

    # 7. Feature importance
    if HAS_XGB:
        importances = model.feature_importances_
    else:
        importances = model.feature_importances_

    importance_data = sorted(
        zip(STAGE1_FEATURES, importances),
        key=lambda x: x[1],
        reverse=True,
    )
    print("\n    Feature Importances:")
    for name, score in importance_data:
        print(f"      {name}: {score:.4f}")

    # 8. Save
    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_report), exist_ok=True)

    joblib.dump(model, args.output_model, compress=3)

    report_payload = {
        "stage": "stage1_human_vs_bot",
        "algorithm": "XGBClassifier" if HAS_XGB else "GradientBoostingClassifier",
        "features": STAGE1_FEATURES,
        "dataset_rows": len(df),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "class_distribution": {
            "human": int((y == 0).sum()),
            "bot": int((y == 1).sum()),
        },
        "optimal_threshold": optimal_threshold,
        "cross_validation": {
            "mean_accuracy": float(cv_results["test_accuracy"].mean()),
            "mean_f1": float(cv_results["test_f1"].mean()),
            "mean_recall": float(cv_results["test_recall"].mean()),
            "mean_roc_auc": float(cv_results["test_roc_auc"].mean()),
        },
        "holdout_metrics": {
            "accuracy": float(accuracy),
            "roc_auc": float(roc_auc),
            "classification_report": report,
            "confusion_matrix": {
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
            },
        },
        "feature_importance": [
            {"feature": n, "importance": float(s)} for n, s in importance_data
        ],
    }

    with open(args.output_report, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    # Save threshold separately for the inference service
    threshold_path = os.path.join(os.path.dirname(args.output_model), "sage_stage1_threshold.json")
    with open(threshold_path, "w") as f:
        json.dump({"optimal_threshold": optimal_threshold}, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Model saved:     {args.output_model}")
    print(f"Report saved:    {args.output_report}")
    print(f"Threshold saved: {threshold_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
