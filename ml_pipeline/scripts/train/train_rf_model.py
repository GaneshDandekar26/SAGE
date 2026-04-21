import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def resolve_base_dir():
    current = Path(__file__).resolve().parent
    for candidate in [current, *current.parents]:
        if candidate.name == "ml_pipeline" and (candidate / "requirements.txt").exists():
            return str(candidate)
    raise RuntimeError("Could not resolve ml_pipeline base directory.")


BASE_DIR = resolve_base_dir()

DATA_DIR = os.path.join(BASE_DIR, "data")
INFERENCE_DIR = os.path.join(BASE_DIR, "inference_service")
MODEL_DIR = os.path.join(INFERENCE_DIR, "models")

MODEL_FILE = os.path.join(MODEL_DIR, "sage_master_rf.pkl")
ENCODER_FILE = os.path.join(MODEL_DIR, "sage_label_encoder.pkl")
FEATURES_FILE = os.path.join(MODEL_DIR, "sage_rf_features.joblib")
REPORT_FILE = os.path.join(INFERENCE_DIR, "evaluation_report.json")

SAGE_FEATURES = [
    "SAGE_Session_Depth",
    "SAGE_Temporal_Variance",
    "SAGE_Request_Velocity",
    "SAGE_Behavioral_Diversity",
]

SOURCE_FILES = [
    "Bot.csv",
    "Brute Force -Web.csv",
    "DDoS attacks-LOIC-HTTP.csv",
    "Infilteration.csv",
]

LABEL_MAP = {
    "Benign": "Benign",
    "Bot": "Bot",
    "Brute Force -Web": "Bot",
    "DDoS attacks-LOIC-HTTP": "Flood",
    "Infilteration": "Infiltration",
    "Infiltration": "Infiltration",
}


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def _compute_sage_features(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip()

    fwd_packets = _safe_numeric(df["Tot Fwd Pkts"])
    bwd_packets = _safe_numeric(df["Tot Bwd Pkts"])
    iat_std = _safe_numeric(df["Flow IAT Std"])
    iat_mean = _safe_numeric(df["Flow IAT Mean"])
    flow_packets_per_sec = _safe_numeric(df["Flow Pkts/s"])
    fwd_packet_len_std = _safe_numeric(df["Fwd Pkt Len Std"])

    out = pd.DataFrame()
    out["SAGE_Session_Depth"] = fwd_packets + bwd_packets

    # Avoid division by zero while preserving temporal ratio meaning.
    out["SAGE_Temporal_Variance"] = iat_std / (iat_mean + 1e-6)
    out["SAGE_Request_Velocity"] = flow_packets_per_sec
    out["SAGE_Behavioral_Diversity"] = fwd_packet_len_std

    out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return out


def _load_multiclass_training_data() -> pd.DataFrame:
    frames = []

    for file_name in SOURCE_FILES:
        file_path = os.path.join(DATA_DIR, file_name)
        print(f"Loading source data: {file_path}")
        df = pd.read_csv(file_path, low_memory=False)
        df.columns = df.columns.str.strip()

        if "Label" not in df.columns:
            raise RuntimeError(f"Missing 'Label' column in {file_name}")

        label_series = df["Label"].astype(str).str.strip()
        mapped_labels = label_series.map(LABEL_MAP)
        keep_mask = mapped_labels.notna()

        selected = df.loc[keep_mask].copy()
        selected["MappedLabel"] = mapped_labels[keep_mask].values
        features = _compute_sage_features(selected)
        features["label"] = selected["MappedLabel"].values
        frames.append(features)

    if not frames:
        raise RuntimeError("No training rows loaded from configured source files.")

    full_df = pd.concat(frames, ignore_index=True)
    full_df = full_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return full_df


def train_sage_model():
    print("Building multiclass dataset from CIC-IDS2018 source files...")
    dataset = _load_multiclass_training_data()

    X = dataset[SAGE_FEATURES]
    y = dataset["label"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded,
    )

    print("Training multiclass RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=16,
        min_samples_leaf=3,
        max_samples=0.6,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    print("\nEvaluating against holdout set...")
    predictions = model.predict(X_test)
    cm = confusion_matrix(y_test, predictions)
    report = classification_report(
        y_test,
        predictions,
        target_names=label_encoder.classes_.tolist(),
        output_dict=True,
    )

    print("\n--- SAGE Multiclass Random Forest Evaluation ---")
    print("Classes:", ", ".join(label_encoder.classes_.tolist()))
    print("Macro F1:", f"{report['macro avg']['f1-score']:.4f}")
    print("Weighted F1:", f"{report['weighted avg']['f1-score']:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_FILE, compress=3)
    joblib.dump(label_encoder, ENCODER_FILE, compress=3)
    joblib.dump(SAGE_FEATURES, FEATURES_FILE, compress=3)

    class_counts = y.value_counts().sort_index().to_dict()
    report_data = {
        "algorithm": "RandomForestClassifier",
        "training_type": "multiclass",
        "classes": label_encoder.classes_.tolist(),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "features_used": SAGE_FEATURES,
        "dataset_rows": int(len(dataset)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "class_distribution": {k: int(v) for k, v in class_counts.items()},
    }
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=4)

    print("\nArtifacts generated:")
    print(f" - Model:   {MODEL_FILE}")
    print(f" - Encoder: {ENCODER_FILE}")
    print(f" - Report:  {REPORT_FILE}")
    print(f" - Classes: {label_encoder.classes_.tolist()}")

if __name__ == "__main__":
    train_sage_model()
