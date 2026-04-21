import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def resolve_base_dir():
    current = Path(__file__).resolve().parent
    for candidate in [current, *current.parents]:
        if candidate.name == "ml_pipeline" and (candidate / "requirements.txt").exists():
            return str(candidate)
    raise RuntimeError("Could not resolve ml_pipeline base directory.")


BASE_DIR = resolve_base_dir()
DEFAULT_DATASET = os.path.join(BASE_DIR, "data", "Bot.csv")
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "evaluation")

RAW_COLUMNS = [
    "Tot Fwd Pkts",
    "Tot Bwd Pkts",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow Pkts/s",
    "Fwd Pkt Len Std",
    "Label",
]

FEATURE_MAP = {
    "SAGE_Session_Depth": "SAGE_Session_Depth",
    "SAGE_Temporal_Variance": "SAGE_Temporal_Variance",
    "SAGE_Request_Velocity": "SAGE_Request_Velocity",
    "SAGE_Behavioral_Diversity": "SAGE_Behavioral_Diversity",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the SAGE Random Forest against CIC-IDS2018 Bot.csv."
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"Path to the CIC-IDS2018 CSV file. Defaults to {DEFAULT_DATASET}",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for generated plots and report files. Defaults to {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row cap for quicker local iteration.",
    )
    return parser.parse_args()


def load_and_transform_dataset(dataset_path, max_rows=None):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path, usecols=RAW_COLUMNS, low_memory=False)

    df["Label"] = df["Label"].astype(str).str.strip().str.lower()
    df = df[df["Label"].isin(["benign", "bot"])].copy()
    if df.empty:
        raise ValueError("Dataset does not contain any rows with Label = Benign or Bot.")

    if max_rows is not None and len(df) > max_rows:
        target_counts = {}
        remaining = max_rows
        labels = list(df["Label"].value_counts().index)
        total_rows = len(df)
        for idx, label in enumerate(labels):
            group_size = int((df["Label"] == label).sum())
            if idx == len(labels) - 1:
                sample_count = remaining
            else:
                sample_count = max(1, round(group_size / total_rows * max_rows))
                sample_count = min(sample_count, group_size, remaining - (len(labels) - idx - 1))
            target_counts[label] = sample_count
            remaining -= sample_count

        sampled_frames = []
        for label, sample_count in target_counts.items():
            sampled_frames.append(
                df[df["Label"] == label].sample(n=sample_count, random_state=42)
            )
        df = pd.concat(sampled_frames, ignore_index=True)

    flow_iat_mean = pd.to_numeric(df["Flow IAT Mean"], errors="coerce")
    flow_iat_std = pd.to_numeric(df["Flow IAT Std"], errors="coerce")

    df["SAGE_Session_Depth"] = (
        pd.to_numeric(df["Tot Fwd Pkts"], errors="coerce").fillna(0)
        + pd.to_numeric(df["Tot Bwd Pkts"], errors="coerce").fillna(0)
    )
    df["SAGE_Temporal_Variance"] = (flow_iat_std / flow_iat_mean.replace(0, np.nan)).fillna(0.0)
    df["SAGE_Request_Velocity"] = pd.to_numeric(df["Flow Pkts/s"], errors="coerce").fillna(0.0)
    df["SAGE_Behavioral_Diversity"] = pd.to_numeric(df["Fwd Pkt Len Std"], errors="coerce").fillna(0.0)
    df["label"] = (df["Label"] == "bot").astype(int)

    features = list(FEATURE_MAP.keys())
    model_df = df[features + ["label"]].replace([float("inf"), float("-inf")], 0.0).dropna()
    if model_df.empty:
        raise ValueError("No valid rows remain after feature engineering.")
    if model_df["label"].nunique() < 2:
        raise ValueError("Evaluation requires both benign and bot samples after preprocessing.")

    X = model_df[features]
    y = model_df["label"]
    return model_df, X, y


def build_pipeline():
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced', n_jobs=-1)),
        ]
    )


def run_cross_validation(pipeline, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    return cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=1)


def plot_confusion_matrix(cm, output_path):
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Reds",
        xticklabels=["Pred Benign", "Pred Bot"],
        yticklabels=["Actual Benign", "Actual Bot"],
    )
    plt.title("SAGE Confusion Matrix on CIC-IDS2018 Bot.csv")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_feature_importance(importances, feature_names, output_path):
    feature_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    feature_df = feature_df.sort_values(by="importance", ascending=False)

    plt.figure(figsize=(9, 5))
    sns.barplot(data=feature_df, x="importance", y="feature", hue="feature", legend=False, palette="viridis")
    plt.title("SAGE Feature Importance on CIC-IDS2018")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return feature_df


def write_report(report_path, report_data):
    with open(report_path, "w", encoding="utf-8") as output_file:
        json.dump(report_data, output_file, indent=2)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading dataset from {args.dataset}")
    model_df, X, y = load_and_transform_dataset(args.dataset, max_rows=args.max_rows)
    print(f"Rows used: {len(model_df)}")
    print(f"Class balance: benign={int((y == 0).sum())}, bot={int((y == 1).sum())}")

    pipeline = build_pipeline()

    print("\nRunning 5-fold cross-validation...")
    cv_results = run_cross_validation(pipeline, X, y)
    for fold_idx, score in enumerate(cv_results["test_accuracy"], start=1):
        print(f"Fold {fold_idx} accuracy: {score:.4f}")

    print(
        f"Mean CV accuracy: {cv_results['test_accuracy'].mean():.4f} "
        f"(std: {cv_results['test_accuracy'].std():.4f})"
    )

    print("\nTraining holdout split for confusion matrix and feature importance...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    holdout_accuracy = accuracy_score(y_test, y_pred)
    holdout_roc_auc = roc_auc_score(y_test, y_proba)
    classification = classification_report(y_test, y_pred, target_names=["Benign", "Bot"], output_dict=True)

    print(f"Holdout accuracy: {holdout_accuracy:.4f}")
    print(f"Holdout ROC-AUC: {holdout_roc_auc:.4f}")
    print("\nConfusion matrix:")
    print(f"True Positives (bots blocked): {tp}")
    print(f"False Positives (benign blocked): {fp}")
    print(f"False Negatives (bots missed): {fn}")
    print(f"True Negatives (benign allowed): {tn}")

    confusion_matrix_path = os.path.join(args.output_dir, "confusion_matrix_bot_csv.png")
    feature_importance_path = os.path.join(args.output_dir, "feature_importance_bot_csv.png")
    report_path = os.path.join(args.output_dir, "evaluation_report_bot_csv.json")

    plot_confusion_matrix(cm, confusion_matrix_path)
    feature_df = plot_feature_importance(
        pipeline.named_steps["model"].feature_importances_,
        list(FEATURE_MAP.keys()),
        feature_importance_path,
    )

    report_data = {
        "dataset_path": os.path.abspath(args.dataset),
        "rows_evaluated": int(len(model_df)),
        "features_used": list(FEATURE_MAP.keys()),
        "feature_derivation": {
            "SAGE_Session_Depth": "Tot Fwd Pkts + Tot Bwd Pkts",
            "SAGE_Temporal_Variance": "Flow IAT Std / Flow IAT Mean",
            "SAGE_Request_Velocity": "Flow Pkts/s",
            "SAGE_Behavioral_Diversity": "Fwd Pkt Len Std",
        },
        "cross_validation": {
            "fold_accuracies": [float(score) for score in cv_results["test_accuracy"]],
            "mean_accuracy": float(cv_results["test_accuracy"].mean()),
            "std_accuracy": float(cv_results["test_accuracy"].std()),
            "mean_precision": float(cv_results["test_precision"].mean()),
            "mean_recall": float(cv_results["test_recall"].mean()),
            "mean_f1": float(cv_results["test_f1"].mean()),
            "mean_roc_auc": float(cv_results["test_roc_auc"].mean()),
        },
        "holdout_metrics": {
            "accuracy": float(holdout_accuracy),
            "roc_auc": float(holdout_roc_auc),
            "classification_report": classification,
            "confusion_matrix": {
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
            },
        },
        "feature_importance": [
            {"feature": row["feature"], "importance": float(row["importance"])}
            for _, row in feature_df.iterrows()
        ],
        "artifacts": {
            "confusion_matrix_plot": os.path.abspath(confusion_matrix_path),
            "feature_importance_plot": os.path.abspath(feature_importance_path),
            "evaluation_report": os.path.abspath(report_path),
        },
    }

    write_report(report_path, report_data)

    print("\nArtifacts written:")
    print(f"- {confusion_matrix_path}")
    print(f"- {feature_importance_path}")
    print(f"- {report_path}")


if __name__ == "__main__":
    main()
