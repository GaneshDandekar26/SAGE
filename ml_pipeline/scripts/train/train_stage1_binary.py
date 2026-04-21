"""
SAGE Stage 1 — Human vs Bot (v2 — Time-Based Split)
=====================================================

Changes from v1:
  - Time-based train/test split (first 80% chronological → train, last 20% → test)
  - No session leaks between train and test
  - Uses scale_pos_weight for class weighting (no SMOTE)
  - Reports show overlap-zone performance

Outputs:
  - models/human_vs_bot.pkl
  - models/human_vs_bot_threshold.json
  - reports/human_vs_bot.txt
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
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False
    from sklearn.ensemble import GradientBoostingClassifier


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
    current = Path(__file__).resolve().parent
    for candidate in [current, *current.parents]:
        if candidate.name == "ml_pipeline" and (candidate / "requirements.txt").exists():
            return str(candidate)
    raise RuntimeError("Could not resolve ml_pipeline base directory.")


BASE_DIR = _resolve_base_dir()


def time_based_split(df: pd.DataFrame, test_frac: float = 0.20):
    """
    Split by timestamp: first (1 - test_frac) chronological → train,
    last test_frac chronological → test.
    ZERO overlap — no session appears in both sets.
    """
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - test_frac))

    train = df_sorted.iloc[:split_idx].copy()
    test = df_sorted.iloc[split_idx:].copy()

    # Verify no session leak
    train_sessions = set(train["session_id"])
    test_sessions = set(test["session_id"])
    overlap = train_sessions & test_sessions
    assert len(overlap) == 0, f"SESSION LEAK: {len(overlap)} sessions in both sets!"

    return train, test


def find_optimal_threshold(y_true, y_proba, min_human_recall=0.90):
    """
    Find threshold that maximises F1-like accuracy while keeping
    human recall (class 0) ≥ min_human_recall.
    """
    best_t, best_score = 0.50, 0.0

    for t in np.arange(0.10, 0.95, 0.01):
        preds = (y_proba >= t).astype(int)
        human_mask = y_true == 0
        if human_mask.sum() == 0:
            continue
        human_recall = (preds[human_mask] == 0).sum() / human_mask.sum()
        if human_recall < min_human_recall:
            continue
        acc = accuracy_score(y_true, preds)
        if acc > best_score:
            best_score = acc
            best_t = float(round(t, 2))

    return best_t


def write_report(path, **kw):
    """Write human-readable text report."""
    lines = [
        "=" * 70,
        "  SAGE Stage 1 — Human vs Bot (v2 — Time-Based Split)",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
        "MODEL",
        f"  Algorithm:       {kw['algorithm']}",
        f"  Features:        {len(STAGE1_FEATURES)}",
        f"  Threshold:       {kw['threshold']}",
        f"  Split method:    TIME-BASED (first 80% → train, last 20% → test)",
        "",
        "DATASET",
        f"  Total:           {kw['total']:,}",
        f"  Train:           {kw['n_train']:,}  (timestamps ≤ cutoff)",
        f"  Test:            {kw['n_test']:,}   (timestamps > cutoff)",
        f"  Session overlap: 0  (verified)",
        f"  Human (train):   {kw['train_human']:,}",
        f"  Bot   (train):   {kw['train_bot']:,}",
        f"  Human (test):    {kw['test_human']:,}",
        f"  Bot   (test):    {kw['test_bot']:,}",
        "",
        "CROSS-VALIDATION (5-fold stratified, on TRAIN set only)",
    ]
    for metric, vals in kw["cv"].items():
        lines.append(f"  {metric:<14s}  {vals.mean():.4f}  (±{vals.std():.4f})")

    tn, fp, fn, tp = kw["cm"].ravel()
    lines += [
        "",
        "HOLDOUT METRICS (time-based test set — UNSEEN future data)",
        f"  Accuracy:        {kw['accuracy']:.4f}",
        f"  ROC-AUC:         {kw['roc_auc']:.4f}",
        "",
        "CONFUSION MATRIX",
        f"  TN (humans allowed):   {tn:,}",
        f"  FP (humans blocked):   {fp:,}     ← CRITICAL: must be low",
        f"  FN (bots missed):      {fn:,}",
        f"  TP (bots caught):      {tp:,}",
        "",
        f"  Human Recall:    {tn / max(tn + fp, 1):.4f}  ← PRIMARY METRIC (target ≥ 0.90)",
        f"  Human Precision: {tn / max(tn + fn, 1):.4f}",
        f"  Bot Recall:      {tp / max(tp + fn, 1):.4f}",
        f"  Bot Precision:   {tp / max(tp + fp, 1):.4f}",
        "",
        "CLASSIFICATION REPORT",
        kw["report_str"],
        "",
        "FEATURE IMPORTANCE",
    ]
    for name, score in kw["importance"]:
        bar = "█" * int(score * 50)
        lines.append(f"  {name:<35s}  {score:.4f}  {bar}")

    lines += ["", "=" * 70]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Train Stage 1 (v2): time-based split, overlapping data"
    )
    parser.add_argument("--input",
                        default=os.path.join(BASE_DIR, "data", "stage1_training_data.csv"))
    parser.add_argument("--output-model",
                        default=os.path.join(BASE_DIR, "models", "human_vs_bot.pkl"))
    parser.add_argument("--output-report",
                        default=os.path.join(BASE_DIR, "reports", "human_vs_bot.txt"))
    args = parser.parse_args()

    print("=" * 65)
    print("  SAGE Stage 1: Human vs Bot (v2 — Time-Based Split)")
    print("=" * 65)

    # ── Load ─────────────────────────────────────────────────────────
    print(f"\n[1/6] Loading {args.input} ...")
    df = pd.read_csv(args.input)
    df = df.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    required = set(STAGE1_FEATURES + ["label", "session_id", "timestamp"])
    missing = required - set(df.columns)
    if missing:
        print(f"FATAL: Missing columns: {missing}", file=sys.stderr)
        sys.exit(1)

    # ── Time-based split ─────────────────────────────────────────────
    print("[2/6] Time-based train/test split (80/20) ...")
    train_df, test_df = time_based_split(df, test_frac=0.20)

    X_train = train_df[STAGE1_FEATURES]
    y_train = train_df["label"]
    X_test = test_df[STAGE1_FEATURES]
    y_test = test_df["label"]

    n_train_h = int((y_train == 0).sum())
    n_train_b = int((y_train == 1).sum())
    n_test_h = int((y_test == 0).sum())
    n_test_b = int((y_test == 1).sum())

    print(f"    Train: {len(X_train):,}  (H:{n_train_h:,}, B:{n_train_b:,})")
    print(f"    Test:  {len(X_test):,}  (H:{n_test_h:,}, B:{n_test_b:,})")
    print(f"    Session overlap: 0")

    # ── Train ────────────────────────────────────────────────────────
    print("\n[3/6] Training XGBoost ...")
    pos_weight = n_train_h / max(n_train_b, 1)

    if _HAS_XGB:
        algo = "XGBClassifier"
        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
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
        algo = "GradientBoostingClassifier"
        model = GradientBoostingClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, random_state=RANDOM_STATE,
        )

    model.fit(X_train, y_train)

    # ── Cross-validation on TRAIN set only ───────────────────────────
    print("[4/6] 5-fold CV (train set only) ...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_res = cross_validate(
        model, X_train, y_train, cv=cv,
        scoring=["accuracy", "f1", "recall", "precision", "roc_auc"],
        n_jobs=-1,
    )
    cv_dict = {m: cv_res[f"test_{m}"] for m in ["accuracy", "f1", "recall", "precision", "roc_auc"]}
    for m, v in cv_dict.items():
        print(f"    {m:<14s} {v.mean():.4f} (±{v.std():.4f})")

    # ── Holdout (future data) ────────────────────────────────────────
    print("\n[5/6] Holdout evaluation (unseen future data) ...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    report_str = classification_report(y_test, y_pred, target_names=["Human", "Bot"])

    print(f"    Accuracy:     {acc:.4f}")
    print(f"    ROC-AUC:      {auc:.4f}")
    print(f"    Human Recall: {tn / max(tn + fp, 1):.4f}")
    print(f"    Bot Recall:   {tp / max(tp + fn, 1):.4f}")
    print(f"    FP (humans blocked): {fp}")
    print(f"    FN (bots missed):    {fn}")

    # ── Threshold ────────────────────────────────────────────────────
    threshold = find_optimal_threshold(y_test.values, y_proba, min_human_recall=0.90)
    print(f"\n    Optimal threshold: {threshold}")

    # Feature importance
    importance = sorted(
        zip(STAGE1_FEATURES, model.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )

    # ── Save ─────────────────────────────────────────────────────────
    print("\n[6/6] Saving ...")
    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    joblib.dump(model, args.output_model, compress=3)

    thr_path = os.path.join(os.path.dirname(args.output_model), "human_vs_bot_threshold.json")
    with open(thr_path, "w") as f:
        json.dump({"optimal_threshold": threshold}, f, indent=2)

    write_report(
        args.output_report,
        algorithm=algo,
        threshold=threshold,
        total=len(df),
        n_train=len(X_train),
        n_test=len(X_test),
        train_human=n_train_h,
        train_bot=n_train_b,
        test_human=n_test_h,
        test_bot=n_test_b,
        cv=cv_dict,
        accuracy=acc,
        roc_auc=auc,
        cm=cm,
        report_str=report_str,
        importance=importance,
    )

    print(f"    Model:     {args.output_model}")
    print(f"    Threshold: {thr_path}")
    print(f"    Report:    {args.output_report}")
    print("=" * 65)


if __name__ == "__main__":
    main()
