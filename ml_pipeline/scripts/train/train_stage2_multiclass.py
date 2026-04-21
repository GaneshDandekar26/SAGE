"""
SAGE Stage 2 — Attack Classifier (v2 — Time-Based Split)
==========================================================

Changes from v1:
  - Time-based train/test split (zero session overlap)
  - class_weight='balanced' (no SMOTE)
  - Trained only on bot data (no human rows)

Outputs:
  - models/attack_classifier.pkl
  - models/attack_classifier_encoder.pkl
  - reports/attack_classifier.txt
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
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder


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


def time_based_split(df: pd.DataFrame, test_frac: float = 0.20):
    """Time-based split: no session overlap."""
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - test_frac))
    train = df_sorted.iloc[:split_idx].copy()
    test = df_sorted.iloc[split_idx:].copy()

    overlap = set(train["session_id"]) & set(test["session_id"])
    assert len(overlap) == 0, f"SESSION LEAK: {len(overlap)} sessions in both sets!"
    return train, test


def write_report(path, **kw):
    lines = [
        "=" * 70,
        "  SAGE Stage 2 — Attack Classifier (v2 — Time-Based Split)",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
        "MODEL",
        "  Algorithm:    RandomForestClassifier",
        f"  Features:     {len(STAGE2_FEATURES)}",
        f"  Classes:      {', '.join(kw['classes'])}",
        f"  Split method: TIME-BASED (first 80% → train, last 20% → test)",
        "",
        "DATASET (Bot-only)",
        f"  Total:   {kw['total']:,}",
        f"  Train:   {kw['n_train']:,}",
        f"  Test:    {kw['n_test']:,}",
        f"  Session overlap: 0  (verified)",
    ]
    for cls, cnt in sorted(kw["class_dist"].items()):
        lines.append(f"  {cls:<12s} {cnt:,}")

    lines += [
        "",
        "CROSS-VALIDATION (5-fold stratified, TRAIN set only)",
    ]
    for metric, vals in kw["cv"].items():
        lines.append(f"  {metric:<20s}  {vals.mean():.4f}  (±{vals.std():.4f})")

    lines += [
        "",
        "HOLDOUT METRICS (time-based test set)",
        f"  Accuracy:  {kw['accuracy']:.4f}",
        "",
        "CONFUSION MATRIX",
    ]
    cm = kw["cm"]
    cls = kw["classes"]
    col_w = max(len(c) for c in cls) + 2
    num_w = max(col_w, len(str(cm.max())) + 2)
    header = " " * (col_w + 2) + "".join(f"{c:>{num_w}}" for c in cls)
    lines.append(header)
    for i, label in enumerate(cls):
        row = f"  {label:<{col_w}}" + "".join(f"{cm[i, j]:>{num_w}}" for j in range(len(cls)))
        lines.append(row)

    lines += [
        "",
        "CLASSIFICATION REPORT",
        kw["report_str"],
        "",
        "FEATURE IMPORTANCE",
    ]
    for name, score in kw["importance"]:
        bar = "█" * int(score * 50)
        lines.append(f"  {name:<35s}  {score:.4f}  {bar}")

    lines += [
        "",
        "GRADUATED RESPONSE MAP",
        "  Flood   → IP BAN",
        "  Scraper → RATE LIMIT",
        "  Recon   → CAPTCHA",
        "",
        "=" * 70,
    ]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Train Stage 2 (v2): time-based split, overlapping data"
    )
    parser.add_argument("--input",
                        default=os.path.join(BASE_DIR, "data", "stage2_training_data.csv"))
    parser.add_argument("--output-model",
                        default=os.path.join(BASE_DIR, "models", "attack_classifier.pkl"))
    parser.add_argument("--output-encoder",
                        default=os.path.join(BASE_DIR, "models", "attack_classifier_encoder.pkl"))
    parser.add_argument("--output-report",
                        default=os.path.join(BASE_DIR, "reports", "attack_classifier.txt"))
    args = parser.parse_args()

    print("=" * 65)
    print("  SAGE Stage 2: Attack Classifier (v2 — Time-Based Split)")
    print("=" * 65)

    # ── Load ─────────────────────────────────────────────────────────
    print(f"\n[1/5] Loading {args.input} ...")
    df = pd.read_csv(args.input)
    df = df.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    if "human" in df["label"].values:
        print("    WARNING: Removing human rows (Stage 2 is bot-only)")
        df = df[df["label"] != "human"].copy()

    X_all = df[STAGE2_FEATURES]
    y_raw = df["label"]
    label_encoder = LabelEncoder()
    y_all = label_encoder.fit_transform(y_raw)
    classes = label_encoder.classes_.tolist()
    class_dist = {k: int(v) for k, v in y_raw.value_counts().to_dict().items()}

    print(f"    Total: {len(df):,}  Classes: {classes}")

    # ── Time-based split ─────────────────────────────────────────────
    print("[2/5] Time-based train/test split (80/20) ...")
    train_df, test_df = time_based_split(df, test_frac=0.20)

    X_train = train_df[STAGE2_FEATURES]
    y_train = label_encoder.transform(train_df["label"])
    X_test = test_df[STAGE2_FEATURES]
    y_test = label_encoder.transform(test_df["label"])

    print(f"    Train: {len(X_train):,}")
    print(f"    Test:  {len(X_test):,}")

    # ── Train ────────────────────────────────────────────────────────
    print("\n[3/5] Training RandomForestClassifier ...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        max_samples=0.8,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # ── CV on train only ─────────────────────────────────────────────
    print("[4/5] 5-fold CV (train set only) ...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_res = cross_validate(
        model, X_train, y_train, cv=cv,
        scoring=["accuracy", "f1_macro", "recall_macro", "precision_macro"],
        n_jobs=-1,
    )
    cv_dict = {m: cv_res[f"test_{m}"] for m in ["accuracy", "f1_macro", "recall_macro", "precision_macro"]}
    for m, v in cv_dict.items():
        print(f"    {m:<20s} {v.mean():.4f} (±{v.std():.4f})")

    # ── Holdout ──────────────────────────────────────────────────────
    print("\n[5/5] Holdout evaluation ...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report_str = classification_report(y_test, y_pred, target_names=classes)

    print(f"    Accuracy: {acc:.4f}")
    for cls_name in classes:
        idx = label_encoder.transform([cls_name])[0]
        tp = cm[idx, idx]
        total = cm[idx].sum()
        print(f"    {cls_name:>8s} recall: {tp / max(total, 1):.4f}  ({tp}/{total})")

    importance = sorted(
        zip(STAGE2_FEATURES, model.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )

    # ── Save ─────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    joblib.dump(model, args.output_model, compress=3)
    joblib.dump(label_encoder, args.output_encoder, compress=3)

    write_report(
        args.output_report,
        classes=classes,
        total=len(df),
        n_train=len(X_train),
        n_test=len(X_test),
        class_dist=class_dist,
        cv=cv_dict,
        accuracy=acc,
        cm=cm,
        report_str=report_str,
        importance=importance,
    )

    print(f"\n    Model:   {args.output_model}")
    print(f"    Encoder: {args.output_encoder}")
    print(f"    Report:  {args.output_report}")
    print("=" * 65)


if __name__ == "__main__":
    main()
