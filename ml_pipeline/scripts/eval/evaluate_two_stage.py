"""
SAGE 2-Stage Pipeline — End-to-End Evaluation
===============================================

Evaluates the full cascaded inference flow with:
  - Per-stage confusion matrices
  - Per-class precision / recall / F1  (focus on Human recall)
  - Failure case analysis (which samples fool the model)
  - System-wide combined metrics
  - Human-readable text report

Output:
  - reports/two_stage_evaluation.txt   — full text report
  - reports/two_stage_evaluation.json  — machine-readable metrics
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
)

try:
    from xgboost import XGBClassifier  # noqa: F401
except ImportError:
    pass


RANDOM_STATE = 99  # Different from training seed (42) to avoid data leakage

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


# ── Synthetic Evaluation Data ────────────────────────────────────────

def _generate_eval_data(n: int) -> pd.DataFrame:
    """
    Generate synthetic evaluation data with 4 true classes.
    Uses RANDOM_STATE=99 (different from training seed=42) to ensure
    no data leakage between training and evaluation sets.
    """
    np.random.seed(RANDOM_STATE)
    frames = []

    # ── HUMAN ────────────────────────────────────────────────────────
    human = {
        "SAGE_InterArrival_CV":       np.random.uniform(0.6, 2.2, n),
        "SAGE_Timing_Entropy":        np.random.uniform(1.6, 2.3, n),
        "SAGE_Pause_Ratio":           np.random.uniform(0.25, 0.65, n),
        "SAGE_Burst_Score":           np.random.exponential(0.012, n).clip(0, 0.08),
        "SAGE_Backtrack_Ratio":       np.random.uniform(0.15, 0.55, n),
        "SAGE_Path_Entropy":          np.random.uniform(2.2, 4.2, n),
        "SAGE_Referral_Chain_Depth":  np.random.uniform(1.8, 3.8, n),
        "SAGE_Session_Depth":         np.random.lognormal(2.5, 0.7, n).clip(3, 150).astype(int),
        "SAGE_Method_Diversity":      np.random.uniform(0.12, 0.35, n),
        "SAGE_Static_Asset_Ratio":    np.random.uniform(0.22, 0.55, n),
        "SAGE_Error_Rate":            np.random.exponential(0.02, n).clip(0, 0.12),
        "SAGE_Payload_Variance":      np.random.uniform(60, 1800, n),
        "SAGE_Request_Velocity":      np.random.uniform(1, 15, n),
        "SAGE_Peak_Burst_RPS":        np.random.uniform(1, 5, n),
        "SAGE_Velocity_Trend":        np.random.normal(0, 1, n),
        "SAGE_Endpoint_Concentration": np.random.uniform(0.1, 0.5, n),
        "SAGE_Cart_Ratio":            np.random.uniform(0.05, 0.3, n),
        "SAGE_Sequential_Traversal":  np.random.exponential(0.05, n).clip(0, 0.2),
        "SAGE_Sensitive_Endpoint_Ratio": np.random.exponential(0.01, n).clip(0, 0.05),
        "SAGE_UA_Entropy":            np.random.uniform(0, 0.5, n),
        "SAGE_Header_Completeness":   np.random.uniform(0.85, 1.0, n),
        "SAGE_Response_Size_Variance": np.random.uniform(100, 3000, n),
    }
    df = pd.DataFrame(human)
    df["true_label"] = "human"
    frames.append(df)

    # ── FLOOD ────────────────────────────────────────────────────────
    flood = {
        "SAGE_InterArrival_CV":       np.random.uniform(0.0, 0.3, n),
        "SAGE_Timing_Entropy":        np.random.uniform(0.0, 0.8, n),
        "SAGE_Pause_Ratio":           np.random.exponential(0.02, n).clip(0, 0.1),
        "SAGE_Burst_Score":           np.random.uniform(0.1, 0.5, n),
        "SAGE_Backtrack_Ratio":       np.random.exponential(0.03, n).clip(0, 0.08),
        "SAGE_Path_Entropy":          np.random.uniform(0.1, 1.2, n),
        "SAGE_Referral_Chain_Depth":  np.random.uniform(1.0, 1.5, n),
        "SAGE_Session_Depth":         np.random.lognormal(4, 1, n).clip(20, 5000).astype(int),
        "SAGE_Method_Diversity":      np.random.exponential(0.01, n).clip(0, 0.05),
        "SAGE_Static_Asset_Ratio":    np.random.exponential(0.01, n).clip(0, 0.04),
        "SAGE_Error_Rate":            np.random.uniform(0.05, 0.3, n),
        "SAGE_Payload_Variance":      np.random.uniform(0, 40, n),
        "SAGE_Request_Velocity":      np.random.uniform(50, 500, n),
        "SAGE_Peak_Burst_RPS":        np.random.uniform(40, 500, n),
        "SAGE_Velocity_Trend":        np.random.normal(0, 2, n),
        "SAGE_Endpoint_Concentration": np.random.uniform(0.88, 1.0, n),
        "SAGE_Cart_Ratio":            np.zeros(n),
        "SAGE_Sequential_Traversal":  np.random.exponential(0.02, n).clip(0, 0.08),
        "SAGE_Sensitive_Endpoint_Ratio": np.random.exponential(0.02, n).clip(0, 0.08),
        "SAGE_UA_Entropy":            np.random.exponential(0.04, n).clip(0, 0.2),
        "SAGE_Header_Completeness":   np.random.uniform(0.0, 0.2, n),
        "SAGE_Response_Size_Variance": np.random.exponential(8, n).clip(0, 40),
    }
    df = pd.DataFrame(flood)
    df["true_label"] = "flood"
    frames.append(df)

    # ── SCRAPER ──────────────────────────────────────────────────────
    scraper = {
        "SAGE_InterArrival_CV":       np.random.uniform(0.05, 0.45, n),
        "SAGE_Timing_Entropy":        np.random.uniform(0.2, 1.0, n),
        "SAGE_Pause_Ratio":           np.random.exponential(0.03, n).clip(0, 0.12),
        "SAGE_Burst_Score":           np.random.uniform(0.05, 0.35, n),
        "SAGE_Backtrack_Ratio":       np.random.exponential(0.04, n).clip(0, 0.12),
        "SAGE_Path_Entropy":          np.random.uniform(0.3, 1.8, n),
        "SAGE_Referral_Chain_Depth":  np.random.uniform(1.0, 1.6, n),
        "SAGE_Session_Depth":         np.random.lognormal(3, 0.8, n).clip(10, 2000).astype(int),
        "SAGE_Method_Diversity":      np.random.exponential(0.02, n).clip(0, 0.08),
        "SAGE_Static_Asset_Ratio":    np.random.exponential(0.015, n).clip(0, 0.06),
        "SAGE_Error_Rate":            np.random.uniform(0.08, 0.35, n),
        "SAGE_Payload_Variance":      np.random.uniform(0, 60, n),
        "SAGE_Request_Velocity":      np.random.uniform(12, 70, n),
        "SAGE_Peak_Burst_RPS":        np.random.uniform(5, 25, n),
        "SAGE_Velocity_Trend":        np.random.normal(-1, 2, n),
        "SAGE_Endpoint_Concentration": np.random.uniform(0.35, 0.65, n),
        "SAGE_Cart_Ratio":            np.random.exponential(0.003, n).clip(0, 0.02),
        "SAGE_Sequential_Traversal":  np.random.uniform(0.45, 0.9, n),
        "SAGE_Sensitive_Endpoint_Ratio": np.random.exponential(0.02, n).clip(0, 0.07),
        "SAGE_UA_Entropy":            np.random.uniform(1.6, 3.2, n),
        "SAGE_Header_Completeness":   np.random.uniform(0.75, 1.0, n),
        "SAGE_Response_Size_Variance": np.random.uniform(250, 4500, n),
    }
    df = pd.DataFrame(scraper)
    df["true_label"] = "scraper"
    frames.append(df)

    # ── RECON ────────────────────────────────────────────────────────
    recon = {
        "SAGE_InterArrival_CV":       np.random.uniform(0.1, 0.5, n),
        "SAGE_Timing_Entropy":        np.random.uniform(0.3, 1.2, n),
        "SAGE_Pause_Ratio":           np.random.exponential(0.04, n).clip(0, 0.15),
        "SAGE_Burst_Score":           np.random.uniform(0.03, 0.25, n),
        "SAGE_Backtrack_Ratio":       np.random.exponential(0.05, n).clip(0, 0.15),
        "SAGE_Path_Entropy":          np.random.uniform(0.5, 2.0, n),
        "SAGE_Referral_Chain_Depth":  np.random.uniform(1.0, 1.8, n),
        "SAGE_Session_Depth":         np.random.lognormal(2.5, 0.6, n).clip(5, 500).astype(int),
        "SAGE_Method_Diversity":      np.random.exponential(0.03, n).clip(0, 0.12),
        "SAGE_Static_Asset_Ratio":    np.random.exponential(0.02, n).clip(0, 0.08),
        "SAGE_Error_Rate":            np.random.uniform(0.1, 0.45, n),
        "SAGE_Payload_Variance":      np.random.uniform(5, 80, n),
        "SAGE_Request_Velocity":      np.random.uniform(3, 22, n),
        "SAGE_Peak_Burst_RPS":        np.random.uniform(1, 8, n),
        "SAGE_Velocity_Trend":        np.random.uniform(3, 12, n),
        "SAGE_Endpoint_Concentration": np.random.uniform(0.08, 0.3, n),
        "SAGE_Cart_Ratio":            np.zeros(n),
        "SAGE_Sequential_Traversal":  np.random.exponential(0.08, n).clip(0, 0.25),
        "SAGE_Sensitive_Endpoint_Ratio": np.random.uniform(0.35, 0.85, n),
        "SAGE_UA_Entropy":            np.random.uniform(0.0, 0.8, n),
        "SAGE_Header_Completeness":   np.random.uniform(0.45, 0.75, n),
        "SAGE_Response_Size_Variance": np.random.uniform(60, 900, n),
    }
    df = pd.DataFrame(recon)
    df["true_label"] = "recon"
    frames.append(df)

    return (
        pd.concat(frames, ignore_index=True)
        .sample(frac=1.0, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )


# ── Helpers ──────────────────────────────────────────────────────────

def _format_cm(cm: np.ndarray, labels: list[str]) -> str:
    """Pretty-print a confusion matrix with row/column labels."""
    col_w = max(len(l) for l in labels) + 2
    num_w = max(max(len(l) for l in labels) + 2, len(str(cm.max())) + 2)

    header = " " * (col_w + 2) + "".join(f"{l:>{num_w}}" for l in labels)
    lines = [header]
    for i, label in enumerate(labels):
        row = f"  {label:<{col_w}}" + "".join(f"{cm[i, j]:>{num_w}}" for j in range(len(labels)))
        lines.append(row)
    return "\n".join(lines)


def _analyze_failures(
    eval_df: pd.DataFrame,
    stage1_pred: np.ndarray,
    stage1_proba: np.ndarray,
    stage2_pred: np.ndarray | None,
    bot_indices: pd.Index,
    features: list[str],
) -> str:
    """
    Analyze misclassified samples to identify failure patterns.
    Returns a text summary.
    """
    lines = []

    # ── Stage 1 failures: humans misclassified as bot ────────────────
    true_binary = (eval_df["true_label"] != "human").astype(int).values
    fp_mask = (true_binary == 0) & (stage1_pred == 1)  # false positives
    fn_mask = (true_binary == 1) & (stage1_pred == 0)  # false negatives

    fp_count = fp_mask.sum()
    fn_count = fn_mask.sum()

    lines.append(f"  Stage 1 False Positives (humans blocked): {fp_count}")
    lines.append(f"  Stage 1 False Negatives (bots missed):    {fn_count}")
    lines.append("")

    if fp_count > 0:
        lines.append("  FALSE POSITIVE ANALYSIS (humans wrongly blocked):")
        fp_df = eval_df[fp_mask]
        fp_proba = stage1_proba[fp_mask]
        lines.append(f"    Count: {fp_count}")
        lines.append(f"    Mean bot_probability: {fp_proba.mean():.4f}")
        lines.append(f"    Max  bot_probability: {fp_proba.max():.4f}")
        lines.append(f"    Min  bot_probability: {fp_proba.min():.4f}")
        lines.append("    Feature means of misclassified humans vs all humans:")
        all_humans = eval_df[eval_df["true_label"] == "human"]
        for feat in STAGE1_FEATURES:
            fp_mean = fp_df[feat].mean()
            all_mean = all_humans[feat].mean()
            diff = fp_mean - all_mean
            if abs(diff) > 0.01:
                direction = "↑" if diff > 0 else "↓"
                lines.append(f"      {feat:<35s}  FP={fp_mean:.4f}  All={all_mean:.4f}  {direction}{abs(diff):.4f}")
        lines.append("")

    if fn_count > 0:
        lines.append("  FALSE NEGATIVE ANALYSIS (bots that slipped through):")
        fn_df = eval_df[fn_mask]
        fn_proba = stage1_proba[fn_mask]
        lines.append(f"    Count: {fn_count}")
        lines.append(f"    Mean bot_probability: {fn_proba.mean():.4f}")
        lines.append(f"    True labels of missed bots:")
        for label, count in fn_df["true_label"].value_counts().items():
            lines.append(f"      {label}: {count}")
        lines.append("    Feature means of missed bots vs all bots:")
        all_bots = eval_df[eval_df["true_label"] != "human"]
        for feat in STAGE1_FEATURES:
            fn_mean = fn_df[feat].mean()
            all_mean = all_bots[feat].mean()
            diff = fn_mean - all_mean
            if abs(diff) > 0.01:
                direction = "↑" if diff > 0 else "↓"
                lines.append(f"      {feat:<35s}  FN={fn_mean:.4f}  All={all_mean:.4f}  {direction}{abs(diff):.4f}")
        lines.append("")

    if fp_count == 0 and fn_count == 0:
        lines.append("  No failure cases detected — Stage 1 achieved perfect separation.")
        lines.append("")

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SAGE 2-stage end-to-end evaluation")
    parser.add_argument("--stage1-model",     default=os.path.join(BASE_DIR, "models", "human_vs_bot.pkl"))
    parser.add_argument("--stage1-threshold", default=os.path.join(BASE_DIR, "models", "human_vs_bot_threshold.json"))
    parser.add_argument("--stage2-model",     default=os.path.join(BASE_DIR, "models", "attack_classifier.pkl"))
    parser.add_argument("--stage2-encoder",   default=os.path.join(BASE_DIR, "models", "attack_classifier_encoder.pkl"))
    parser.add_argument("--eval-samples",     type=int, default=5000, help="Samples per class")
    parser.add_argument("--output-txt",       default=os.path.join(BASE_DIR, "reports", "two_stage_evaluation.txt"))
    parser.add_argument("--output-json",      default=os.path.join(BASE_DIR, "reports", "two_stage_evaluation.json"))
    args = parser.parse_args()

    print("=" * 65)
    print("  SAGE 2-Stage Pipeline — End-to-End Evaluation")
    print("=" * 65)

    # ── Load models ──────────────────────────────────────────────────
    print("\n[1/5] Loading models ...")
    s1_model = joblib.load(args.stage1_model)
    s2_model = joblib.load(args.stage2_model)
    s2_encoder = joblib.load(args.stage2_encoder)

    threshold = 0.50
    if os.path.exists(args.stage1_threshold):
        with open(args.stage1_threshold) as f:
            threshold = json.load(f).get("optimal_threshold", 0.50)
    print(f"    Stage 1 threshold: {threshold}")
    print(f"    Stage 2 classes:   {s2_encoder.classes_.tolist()}")

    # ── Generate data ────────────────────────────────────────────────
    print(f"\n[2/5] Generating {args.eval_samples} samples × 4 classes ...")
    eval_df = _generate_eval_data(args.eval_samples)
    N = len(eval_df)
    print(f"    Total: {N:,}")

    # ── Stage 1 ──────────────────────────────────────────────────────
    print("\n[3/5] Stage 1: Human vs Bot ...")
    X1 = eval_df[STAGE1_FEATURES].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    s1_proba = s1_model.predict_proba(X1)[:, 1]
    s1_pred = (s1_proba >= threshold).astype(int)

    true_binary = (eval_df["true_label"] != "human").astype(int)
    s1_acc = accuracy_score(true_binary, s1_pred)
    s1_cm = confusion_matrix(true_binary, s1_pred)
    s1_report = classification_report(true_binary, s1_pred, target_names=["Human", "Bot"], output_dict=True)
    s1_report_str = classification_report(true_binary, s1_pred, target_names=["Human", "Bot"])
    tn, fp, fn, tp = s1_cm.ravel()

    print(f"    Accuracy:     {s1_acc:.4f}")
    print(f"    Human Recall: {s1_report['Human']['recall']:.4f}  (target: ≥ 0.90)")
    print(f"    Bot Recall:   {s1_report['Bot']['recall']:.4f}")
    print(f"    FP (humans blocked): {fp}")
    print(f"    FN (bots missed):    {fn}")

    # ── Stage 2 ──────────────────────────────────────────────────────
    print("\n[4/5] Stage 2: Classifying caught bots ...")
    bot_mask = s1_pred == 1
    bot_idx = eval_df.index[bot_mask]

    s2_pred_labels = np.array([""] * N, dtype=object)
    s2_accuracy = 0.0
    s2_report_str = ""
    s2_cm = np.array([])
    s2_report = {}

    if bot_mask.sum() > 0:
        X2 = eval_df.loc[bot_idx, STAGE2_FEATURES].replace([np.inf, -np.inf], 0.0).fillna(0.0)
        s2_enc_pred = s2_model.predict(X2)
        s2_labels = s2_encoder.inverse_transform(s2_enc_pred)
        s2_pred_labels[bot_idx] = s2_labels

        # Evaluate only on actual bots (not false-positive humans)
        true_bot_labels = eval_df.loc[bot_idx, "true_label"].values
        actual_bot = true_bot_labels != "human"
        if actual_bot.sum() > 0:
            s2_accuracy = accuracy_score(true_bot_labels[actual_bot], s2_labels[actual_bot])
            s2_report = classification_report(true_bot_labels[actual_bot], s2_labels[actual_bot], output_dict=True)
            s2_report_str = classification_report(true_bot_labels[actual_bot], s2_labels[actual_bot])
            s2_cm = confusion_matrix(true_bot_labels[actual_bot], s2_labels[actual_bot],
                                     labels=s2_encoder.classes_.tolist())
            print(f"    Accuracy: {s2_accuracy:.4f}")
            for cls in s2_encoder.classes_:
                if cls in s2_report:
                    r = s2_report[cls]
                    print(f"    {cls:>8s}  P={r['precision']:.4f}  R={r['recall']:.4f}  F1={r['f1-score']:.4f}")

    # ── Combined system ──────────────────────────────────────────────
    print("\n[5/5] Combined system metrics ...")
    final = []
    for i in range(N):
        if s1_pred[i] == 0:
            final.append("human")
        else:
            final.append(s2_pred_labels[i] if s2_pred_labels[i] else "unknown_bot")

    sys_labels = ["human", "flood", "scraper", "recon"]
    sys_acc = accuracy_score(eval_df["true_label"], final)
    sys_report = classification_report(eval_df["true_label"], final, output_dict=True)
    sys_report_str = classification_report(eval_df["true_label"], final)
    sys_cm = confusion_matrix(eval_df["true_label"], final, labels=sys_labels)
    fpr = fp / max(tn + fp, 1)

    print(f"    System Accuracy:      {sys_acc:.4f}")
    print(f"    System Macro F1:      {sys_report['macro avg']['f1-score']:.4f}")
    print(f"    Human FP Rate:        {fpr:.4f}")
    for cls in sys_labels:
        if cls in sys_report:
            r = sys_report[cls]
            print(f"    {cls:>8s}  P={r['precision']:.4f}  R={r['recall']:.4f}  F1={r['f1-score']:.4f}")

    # ── Failure analysis ─────────────────────────────────────────────
    failure_text = _analyze_failures(eval_df, s1_pred, s1_proba, s2_pred_labels, bot_idx, STAGE1_FEATURES)

    # ── Write text report ────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output_txt), exist_ok=True)

    report_lines = [
        "=" * 70,
        "  SAGE 2-Stage Pipeline — End-to-End Evaluation Report",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Evaluation seed: {RANDOM_STATE} (different from training seed 42)",
        "=" * 70,
        "",
        "CONFIGURATION",
        f"  Stage 1 threshold:   {threshold}",
        f"  Eval samples/class:  {args.eval_samples:,}",
        f"  Total samples:       {N:,}",
        "",
        "─" * 70,
        "STAGE 1: HUMAN vs BOT (Binary)",
        "─" * 70,
        "",
        "  CONFUSION MATRIX",
        _format_cm(s1_cm, ["Human", "Bot"]),
        "",
        f"  Accuracy:           {s1_acc:.4f}",
        f"  Human Recall:       {s1_report['Human']['recall']:.4f}  ← PRIMARY METRIC",
        f"  Human Precision:    {s1_report['Human']['precision']:.4f}",
        f"  Bot Recall:         {s1_report['Bot']['recall']:.4f}",
        f"  Bot Precision:      {s1_report['Bot']['precision']:.4f}",
        "",
        f"  Humans allowed:     {tn:,}",
        f"  Humans blocked (FP):{fp:,}  ← MUST BE ZERO OR NEAR-ZERO",
        f"  Bots missed (FN):   {fn:,}",
        f"  Bots caught:        {tp:,}",
        "",
        "  CLASSIFICATION REPORT",
        s1_report_str,
        "",
        "─" * 70,
        "STAGE 2: ATTACK CLASSIFICATION (3-class, bot-only)",
        "─" * 70,
        "",
    ]

    if s2_cm.size > 0:
        report_lines += [
            "  CONFUSION MATRIX",
            _format_cm(s2_cm, s2_encoder.classes_.tolist()),
            "",
            f"  Accuracy:  {s2_accuracy:.4f}",
            "",
            "  CLASSIFICATION REPORT",
            s2_report_str,
        ]
    else:
        report_lines.append("  (No bots detected — Stage 2 not evaluated)")

    report_lines += [
        "",
        "─" * 70,
        "COMBINED SYSTEM (4-class end-to-end)",
        "─" * 70,
        "",
        "  CONFUSION MATRIX (rows=true, cols=predicted)",
        _format_cm(sys_cm, sys_labels),
        "",
        f"  System Accuracy:    {sys_acc:.4f}",
        f"  Human FP Rate:      {fpr:.4f}",
        f"  System Macro F1:    {sys_report['macro avg']['f1-score']:.4f}",
        f"  System Weighted F1: {sys_report['weighted avg']['f1-score']:.4f}",
        "",
        "  PER-CLASS BREAKDOWN",
    ]
    for cls in sys_labels:
        if cls in sys_report:
            r = sys_report[cls]
            report_lines.append(
                f"    {cls:<10s}  Precision={r['precision']:.4f}  "
                f"Recall={r['recall']:.4f}  F1={r['f1-score']:.4f}  "
                f"Support={int(r['support'])}"
            )

    report_lines += [
        "",
        "  CLASSIFICATION REPORT",
        sys_report_str,
        "",
        "─" * 70,
        "FAILURE CASE ANALYSIS",
        "─" * 70,
        "",
        failure_text,
        "─" * 70,
        "THRESHOLD TUNING GUIDE",
        "─" * 70,
        "",
        "  Current threshold: " + str(threshold),
        "",
        "  To INCREASE Human recall (fewer humans blocked):",
        "    → Raise threshold (e.g. 0.60, 0.70, 0.80)",
        "    → Trade-off: more bots may slip through",
        "",
        "  To CATCH MORE BOTS (fewer bots missed):",
        "    → Lower threshold (e.g. 0.30, 0.20, 0.10)",
        "    → Trade-off: more humans may be falsely blocked",
        "",
        "  How to change:",
        "    1. Runtime:  PUT http://localhost:8000/config/threshold",
        '       Body:    {"threshold": 0.40}',
        "    2. Env var:  export SAGE_BOT_THRESHOLD=0.40",
        "    3. File:     models/human_vs_bot_threshold.json",
        "",
        "=" * 70,
    ]

    with open(args.output_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    # ── Write JSON report ────────────────────────────────────────────
    json_data = {
        "generated": datetime.now().isoformat(),
        "eval_seed": RANDOM_STATE,
        "threshold": threshold,
        "total_samples": N,
        "stage1": {
            "accuracy": float(s1_acc),
            "human_recall": float(s1_report["Human"]["recall"]),
            "bot_recall": float(s1_report["Bot"]["recall"]),
            "human_fp_rate": float(fpr),
            "confusion_matrix": s1_cm.tolist(),
            "classification_report": s1_report,
        },
        "stage2": {
            "accuracy": float(s2_accuracy),
            "confusion_matrix": s2_cm.tolist() if hasattr(s2_cm, "tolist") else [],
            "classification_report": s2_report,
        },
        "system": {
            "accuracy": float(sys_acc),
            "macro_f1": float(sys_report["macro avg"]["f1-score"]),
            "confusion_matrix": sys_cm.tolist(),
            "classification_report": sys_report,
        },
        "failure_analysis": {
            "false_positives": int(fp),
            "false_negatives": int(fn),
        },
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)

    print(f"\n  Reports saved:")
    print(f"    {args.output_txt}")
    print(f"    {args.output_json}")
    print("=" * 65)


if __name__ == "__main__":
    main()
