# ml_pipeline/scripts/eval/evaluate_two_stage.py
"""
End-to-End Evaluation of the SAGE 2-Stage ML Pipeline

Simulates the full cascaded inference flow:
    1. Stage 1 (Human vs Bot) filters all traffic
    2. Stage 2 (Flood/Scraper/Recon) classifies confirmed bots

Produces:
    - Overall system accuracy
    - Per-stage metrics (precision, recall, F1)
    - Human safety metrics (false positive rate)
    - Confusion matrices for both stages
    - Combined system report JSON
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
)

try:
    from xgboost import XGBClassifier
except ImportError:
    pass


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

# Combined features = union of Stage 1 + Stage 2 (22 total, some overlap)
ALL_FEATURES = list(dict.fromkeys(STAGE1_FEATURES + STAGE2_FEATURES))


def _synthesize_evaluation_data(n_per_class: int = 5000) -> pd.DataFrame:
    """
    Generate a synthetic evaluation dataset with 4 true classes:
    human, flood, scraper, recon.  Each row has ALL 22 features so
    both stages can be evaluated.
    """
    np.random.seed(99)  # Different seed from training for honest eval
    frames = []

    # === HUMAN sessions ===
    n = n_per_class
    human = {
        # Stage 1 features (behavioural)
        "SAGE_InterArrival_CV": np.random.uniform(0.6, 2.2, n),
        "SAGE_Timing_Entropy": np.random.uniform(1.6, 2.3, n),
        "SAGE_Pause_Ratio": np.random.uniform(0.25, 0.65, n),
        "SAGE_Burst_Score": np.random.exponential(0.012, n).clip(0, 0.08),
        "SAGE_Backtrack_Ratio": np.random.uniform(0.15, 0.55, n),
        "SAGE_Path_Entropy": np.random.uniform(2.2, 4.2, n),
        "SAGE_Referral_Chain_Depth": np.random.uniform(1.8, 3.8, n),
        "SAGE_Session_Depth": np.random.lognormal(2.5, 0.7, n).clip(3, 150).astype(int),
        "SAGE_Method_Diversity": np.random.uniform(0.12, 0.35, n),
        "SAGE_Static_Asset_Ratio": np.random.uniform(0.22, 0.55, n),
        "SAGE_Error_Rate": np.random.exponential(0.02, n).clip(0, 0.12),
        "SAGE_Payload_Variance": np.random.uniform(60, 1800, n),
        # Stage 2 features (less meaningful for humans but need values)
        "SAGE_Request_Velocity": np.random.uniform(1, 15, n),
        "SAGE_Peak_Burst_RPS": np.random.uniform(1, 5, n),
        "SAGE_Velocity_Trend": np.random.normal(0, 1, n),
        "SAGE_Endpoint_Concentration": np.random.uniform(0.1, 0.5, n),
        "SAGE_Cart_Ratio": np.random.uniform(0.05, 0.3, n),
        "SAGE_Sequential_Traversal": np.random.exponential(0.05, n).clip(0, 0.2),
        "SAGE_Sensitive_Endpoint_Ratio": np.random.exponential(0.01, n).clip(0, 0.05),
        "SAGE_UA_Entropy": np.random.uniform(0, 0.5, n),
        "SAGE_Header_Completeness": np.random.uniform(0.85, 1.0, n),
        "SAGE_Response_Size_Variance": np.random.uniform(100, 3000, n),
    }
    df_human = pd.DataFrame(human)
    df_human["true_label"] = "human"
    frames.append(df_human)

    # === FLOOD bots ===
    flood = {
        "SAGE_InterArrival_CV": np.random.uniform(0.0, 0.3, n),
        "SAGE_Timing_Entropy": np.random.uniform(0.0, 0.8, n),
        "SAGE_Pause_Ratio": np.random.exponential(0.02, n).clip(0, 0.1),
        "SAGE_Burst_Score": np.random.uniform(0.1, 0.5, n),
        "SAGE_Backtrack_Ratio": np.random.exponential(0.03, n).clip(0, 0.08),
        "SAGE_Path_Entropy": np.random.uniform(0.1, 1.2, n),
        "SAGE_Referral_Chain_Depth": np.random.uniform(1.0, 1.5, n),
        "SAGE_Session_Depth": np.random.lognormal(4, 1, n).clip(20, 5000).astype(int),
        "SAGE_Method_Diversity": np.random.exponential(0.01, n).clip(0, 0.05),
        "SAGE_Static_Asset_Ratio": np.random.exponential(0.01, n).clip(0, 0.04),
        "SAGE_Error_Rate": np.random.uniform(0.05, 0.3, n),
        "SAGE_Payload_Variance": np.random.uniform(0, 40, n),
        "SAGE_Request_Velocity": np.random.uniform(50, 500, n),
        "SAGE_Peak_Burst_RPS": np.random.uniform(40, 500, n),
        "SAGE_Velocity_Trend": np.random.normal(0, 2, n),
        "SAGE_Endpoint_Concentration": np.random.uniform(0.88, 1.0, n),
        "SAGE_Cart_Ratio": np.zeros(n),
        "SAGE_Sequential_Traversal": np.random.exponential(0.02, n).clip(0, 0.08),
        "SAGE_Sensitive_Endpoint_Ratio": np.random.exponential(0.02, n).clip(0, 0.08),
        "SAGE_UA_Entropy": np.random.exponential(0.04, n).clip(0, 0.2),
        "SAGE_Header_Completeness": np.random.uniform(0.0, 0.2, n),
        "SAGE_Response_Size_Variance": np.random.exponential(8, n).clip(0, 40),
    }
    df_flood = pd.DataFrame(flood)
    df_flood["true_label"] = "flood"
    frames.append(df_flood)

    # === SCRAPER bots ===
    scraper = {
        "SAGE_InterArrival_CV": np.random.uniform(0.05, 0.45, n),
        "SAGE_Timing_Entropy": np.random.uniform(0.2, 1.0, n),
        "SAGE_Pause_Ratio": np.random.exponential(0.03, n).clip(0, 0.12),
        "SAGE_Burst_Score": np.random.uniform(0.05, 0.35, n),
        "SAGE_Backtrack_Ratio": np.random.exponential(0.04, n).clip(0, 0.12),
        "SAGE_Path_Entropy": np.random.uniform(0.3, 1.8, n),
        "SAGE_Referral_Chain_Depth": np.random.uniform(1.0, 1.6, n),
        "SAGE_Session_Depth": np.random.lognormal(3, 0.8, n).clip(10, 2000).astype(int),
        "SAGE_Method_Diversity": np.random.exponential(0.02, n).clip(0, 0.08),
        "SAGE_Static_Asset_Ratio": np.random.exponential(0.015, n).clip(0, 0.06),
        "SAGE_Error_Rate": np.random.uniform(0.08, 0.35, n),
        "SAGE_Payload_Variance": np.random.uniform(0, 60, n),
        "SAGE_Request_Velocity": np.random.uniform(12, 70, n),
        "SAGE_Peak_Burst_RPS": np.random.uniform(5, 25, n),
        "SAGE_Velocity_Trend": np.random.normal(-1, 2, n),
        "SAGE_Endpoint_Concentration": np.random.uniform(0.35, 0.65, n),
        "SAGE_Cart_Ratio": np.random.exponential(0.003, n).clip(0, 0.02),
        "SAGE_Sequential_Traversal": np.random.uniform(0.45, 0.9, n),
        "SAGE_Sensitive_Endpoint_Ratio": np.random.exponential(0.02, n).clip(0, 0.07),
        "SAGE_UA_Entropy": np.random.uniform(1.6, 3.2, n),
        "SAGE_Header_Completeness": np.random.uniform(0.75, 1.0, n),
        "SAGE_Response_Size_Variance": np.random.uniform(250, 4500, n),
    }
    df_scraper = pd.DataFrame(scraper)
    df_scraper["true_label"] = "scraper"
    frames.append(df_scraper)

    # === RECON probes ===
    recon = {
        "SAGE_InterArrival_CV": np.random.uniform(0.1, 0.5, n),
        "SAGE_Timing_Entropy": np.random.uniform(0.3, 1.2, n),
        "SAGE_Pause_Ratio": np.random.exponential(0.04, n).clip(0, 0.15),
        "SAGE_Burst_Score": np.random.uniform(0.03, 0.25, n),
        "SAGE_Backtrack_Ratio": np.random.exponential(0.05, n).clip(0, 0.15),
        "SAGE_Path_Entropy": np.random.uniform(0.5, 2.0, n),
        "SAGE_Referral_Chain_Depth": np.random.uniform(1.0, 1.8, n),
        "SAGE_Session_Depth": np.random.lognormal(2.5, 0.6, n).clip(5, 500).astype(int),
        "SAGE_Method_Diversity": np.random.exponential(0.03, n).clip(0, 0.12),
        "SAGE_Static_Asset_Ratio": np.random.exponential(0.02, n).clip(0, 0.08),
        "SAGE_Error_Rate": np.random.uniform(0.1, 0.45, n),
        "SAGE_Payload_Variance": np.random.uniform(5, 80, n),
        "SAGE_Request_Velocity": np.random.uniform(3, 22, n),
        "SAGE_Peak_Burst_RPS": np.random.uniform(1, 8, n),
        "SAGE_Velocity_Trend": np.random.uniform(3, 12, n),
        "SAGE_Endpoint_Concentration": np.random.uniform(0.08, 0.3, n),
        "SAGE_Cart_Ratio": np.zeros(n),
        "SAGE_Sequential_Traversal": np.random.exponential(0.08, n).clip(0, 0.25),
        "SAGE_Sensitive_Endpoint_Ratio": np.random.uniform(0.35, 0.85, n),
        "SAGE_UA_Entropy": np.random.uniform(0.0, 0.8, n),
        "SAGE_Header_Completeness": np.random.uniform(0.45, 0.75, n),
        "SAGE_Response_Size_Variance": np.random.uniform(60, 900, n),
    }
    df_recon = pd.DataFrame(recon)
    df_recon["true_label"] = "recon"
    frames.append(df_recon)

    return pd.concat(frames, ignore_index=True).sample(frac=1.0, random_state=99).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Evaluate SAGE 2-stage pipeline end-to-end")
    parser.add_argument(
        "--stage1-model",
        default=os.path.join(BASE_DIR, "models", "sage_stage1_human_vs_bot.pkl"),
    )
    parser.add_argument(
        "--stage1-threshold",
        default=os.path.join(BASE_DIR, "models", "sage_stage1_threshold.json"),
    )
    parser.add_argument(
        "--stage2-model",
        default=os.path.join(BASE_DIR, "models", "sage_stage2_bot_classifier.pkl"),
    )
    parser.add_argument(
        "--stage2-encoder",
        default=os.path.join(BASE_DIR, "models", "sage_stage2_label_encoder.pkl"),
    )
    parser.add_argument(
        "--eval-samples", type=int, default=5000,
        help="Number of evaluation samples per class (human/flood/scraper/recon)",
    )
    parser.add_argument(
        "--output-report",
        default=os.path.join(BASE_DIR, "reports", "two_stage_evaluation_report.json"),
    )
    args = parser.parse_args()

    print("=" * 60)
    print("SAGE 2-Stage Pipeline — End-to-End Evaluation")
    print("=" * 60)

    # 1. Load models
    print("\n[1] Loading models...")
    stage1_model = joblib.load(args.stage1_model)
    stage2_model = joblib.load(args.stage2_model)
    stage2_encoder = joblib.load(args.stage2_encoder)

    threshold = 0.5
    if os.path.exists(args.stage1_threshold):
        with open(args.stage1_threshold) as f:
            threshold = json.load(f).get("optimal_threshold", 0.5)
    print(f"    Stage 1 threshold: {threshold}")
    print(f"    Stage 2 classes: {stage2_encoder.classes_.tolist()}")

    # 2. Generate evaluation data
    print(f"\n[2] Generating {args.eval_samples} samples per class...")
    eval_df = _synthesize_evaluation_data(n_per_class=args.eval_samples)
    print(f"    Total evaluation samples: {len(eval_df)}")
    print(f"    Distribution:\n{eval_df['true_label'].value_counts().to_string()}")

    # 3. Stage 1: Human vs Bot
    print("\n[3] Stage 1: Human vs Bot classification...")
    X_stage1 = eval_df[STAGE1_FEATURES].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    stage1_proba = stage1_model.predict_proba(X_stage1)[:, 1]
    stage1_pred = (stage1_proba >= threshold).astype(int)  # 0=human, 1=bot

    # True binary labels: human=0, everything else=1
    true_binary = (eval_df["true_label"] != "human").astype(int)

    stage1_accuracy = accuracy_score(true_binary, stage1_pred)
    stage1_cm = confusion_matrix(true_binary, stage1_pred)
    stage1_report = classification_report(
        true_binary, stage1_pred,
        target_names=["Human", "Bot"],
        output_dict=True,
    )

    tn, fp, fn, tp = stage1_cm.ravel()
    print(f"    Stage 1 Accuracy: {stage1_accuracy:.4f}")
    print(f"    Human Recall: {stage1_report['Human']['recall']:.4f}")
    print(f"    Bot Recall:   {stage1_report['Bot']['recall']:.4f}")
    print(f"    Humans correctly allowed: {tn}")
    print(f"    Humans incorrectly blocked (FP): {fp}")
    print(f"    Bots missed (FN): {fn}")
    print(f"    Bots correctly caught: {tp}")

    # 4. Stage 2: Classify bots that were caught
    print("\n[4] Stage 2: Classifying caught bots...")
    bot_mask = stage1_pred == 1
    bot_indices = eval_df.index[bot_mask]

    if len(bot_indices) > 0:
        X_stage2 = eval_df.loc[bot_indices, STAGE2_FEATURES].replace([np.inf, -np.inf], 0.0).fillna(0.0)
        stage2_pred_encoded = stage2_model.predict(X_stage2)
        stage2_pred_labels = stage2_encoder.inverse_transform(stage2_pred_encoded)

        # True labels for bots only (excluding humans that were incorrectly flagged)
        true_bot_labels = eval_df.loc[bot_indices, "true_label"].values

        # For the bots that are actually bots (not false positive humans)
        actual_bot_mask = true_bot_labels != "human"
        if actual_bot_mask.sum() > 0:
            stage2_accuracy = accuracy_score(
                true_bot_labels[actual_bot_mask],
                stage2_pred_labels[actual_bot_mask],
            )
            stage2_report = classification_report(
                true_bot_labels[actual_bot_mask],
                stage2_pred_labels[actual_bot_mask],
                output_dict=True,
            )
            stage2_cm = confusion_matrix(
                true_bot_labels[actual_bot_mask],
                stage2_pred_labels[actual_bot_mask],
            )
            print(f"    Stage 2 Accuracy (on true bots): {stage2_accuracy:.4f}")
            print(f"    Stage 2 Macro F1: {stage2_report['macro avg']['f1-score']:.4f}")
            for cls in ["flood", "scraper", "recon"]:
                if cls in stage2_report:
                    print(f"    {cls} — P: {stage2_report[cls]['precision']:.4f}  "
                          f"R: {stage2_report[cls]['recall']:.4f}  "
                          f"F1: {stage2_report[cls]['f1-score']:.4f}")
        else:
            stage2_accuracy = 0.0
            stage2_report = {}
            stage2_cm = []
            print("    No true bots in Stage 2 evaluation set!")
    else:
        stage2_accuracy = 0.0
        stage2_report = {}
        stage2_cm = []
        print("    Stage 1 didn't identify any bots!")

    # 5. Combined system metrics
    print("\n[5] Combined System Metrics...")

    # Build final predictions for all 4 classes
    final_predictions = []
    for i in range(len(eval_df)):
        if stage1_pred[i] == 0:
            final_predictions.append("human")
        elif i in bot_indices:
            idx_in_stage2 = list(bot_indices).index(i)
            final_predictions.append(stage2_pred_labels[idx_in_stage2])
        else:
            final_predictions.append("unknown_bot")

    eval_df["predicted_label"] = final_predictions
    true_labels = eval_df["true_label"].values
    pred_labels = eval_df["predicted_label"].values

    system_accuracy = accuracy_score(true_labels, pred_labels)
    system_report = classification_report(
        true_labels, pred_labels, output_dict=True
    )
    system_cm = confusion_matrix(
        true_labels, pred_labels,
        labels=["human", "flood", "scraper", "recon"],
    )

    print(f"    System Accuracy:      {system_accuracy:.4f}")
    print(f"    System Macro F1:      {system_report['macro avg']['f1-score']:.4f}")
    print(f"    System Weighted F1:   {system_report['weighted avg']['f1-score']:.4f}")
    print(f"    Human False Positive Rate: {fp / (tn + fp):.4f}" if (tn + fp) > 0 else "")

    for cls in ["human", "flood", "scraper", "recon"]:
        if cls in system_report:
            print(f"    {cls} — P: {system_report[cls]['precision']:.4f}  "
                  f"R: {system_report[cls]['recall']:.4f}  "
                  f"F1: {system_report[cls]['f1-score']:.4f}")

    print(f"\n    System Confusion Matrix (human/flood/scraper/recon):")
    print(f"    {system_cm}")

    # 6. Save report
    os.makedirs(os.path.dirname(args.output_report), exist_ok=True)
    report_data = {
        "evaluation_type": "two_stage_end_to_end",
        "eval_samples_per_class": args.eval_samples,
        "total_samples": len(eval_df),
        "stage1_threshold": threshold,
        "stage1_metrics": {
            "accuracy": float(stage1_accuracy),
            "classification_report": stage1_report,
            "confusion_matrix": stage1_cm.tolist(),
        },
        "stage2_metrics": {
            "accuracy": float(stage2_accuracy),
            "classification_report": stage2_report,
            "confusion_matrix": stage2_cm.tolist() if hasattr(stage2_cm, "tolist") else stage2_cm,
        },
        "system_metrics": {
            "accuracy": float(system_accuracy),
            "classification_report": system_report,
            "confusion_matrix": system_cm.tolist(),
            "human_false_positive_rate": float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0,
        },
    }

    with open(args.output_report, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Evaluation report saved: {args.output_report}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
