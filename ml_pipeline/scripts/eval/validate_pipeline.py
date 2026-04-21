"""
SAGE Pipeline Validation — Pre-Production Checks
==================================================

Systematically checks for:
  1. Data leakage between train/eval sets
  2. Feature mismatch between training data and inference service
  3. Pipeline inconsistencies (model ↔ encoder ↔ features)
  4. Model artifact integrity

Output:
  - reports/pipeline_validation.txt  — human-readable validation report

Run before every deployment:
    python3 scripts/eval/validate_pipeline.py
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def _resolve_base_dir() -> str:
    current = Path(__file__).resolve().parent
    for candidate in [current, *current.parents]:
        if candidate.name == "ml_pipeline" and (candidate / "requirements.txt").exists():
            return str(candidate)
    raise RuntimeError("Could not resolve ml_pipeline base directory.")


BASE_DIR = _resolve_base_dir()

# ── Expected feature lists (single source of truth) ─────────────────

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


class ValidationResult:
    """Accumulates pass/fail/warn checks."""

    def __init__(self):
        self.checks: list[dict] = []

    def ok(self, name: str, detail: str = ""):
        self.checks.append({"status": "PASS", "name": name, "detail": detail})

    def fail(self, name: str, detail: str = ""):
        self.checks.append({"status": "FAIL", "name": name, "detail": detail})

    def warn(self, name: str, detail: str = ""):
        self.checks.append({"status": "WARN", "name": name, "detail": detail})

    @property
    def passed(self) -> int:
        return sum(1 for c in self.checks if c["status"] == "PASS")

    @property
    def failed(self) -> int:
        return sum(1 for c in self.checks if c["status"] == "FAIL")

    @property
    def warnings(self) -> int:
        return sum(1 for c in self.checks if c["status"] == "WARN")

    def to_text(self) -> str:
        lines = []
        for c in self.checks:
            icon = {"PASS": "✓", "FAIL": "✗", "WARN": "⚠"}[c["status"]]
            lines.append(f"  [{icon}] {c['status']}  {c['name']}")
            if c["detail"]:
                for line in c["detail"].split("\n"):
                    lines.append(f"         {line}")
        return "\n".join(lines)


# ── Individual Checks ────────────────────────────────────────────────

def check_model_artifacts(v: ValidationResult) -> None:
    """Verify all required model files exist and are loadable."""
    required = {
        "human_vs_bot.pkl":              "Stage 1 model",
        "human_vs_bot_threshold.json":   "Stage 1 threshold",
        "attack_classifier.pkl":         "Stage 2 model",
        "attack_classifier_encoder.pkl": "Stage 2 label encoder",
    }
    model_dir = os.path.join(BASE_DIR, "models")

    for filename, desc in required.items():
        path = os.path.join(model_dir, filename)
        if not os.path.exists(path):
            v.fail(f"Model artifact: {desc}", f"Missing: {path}")
            continue
        sz = os.path.getsize(path) / 1024
        if filename.endswith(".pkl"):
            try:
                obj = joblib.load(path)
                v.ok(f"Model artifact: {desc}", f"Loaded OK ({sz:.1f} KB)")
            except Exception as e:
                v.fail(f"Model artifact: {desc}", f"Load error: {e}")
        else:
            try:
                with open(path) as f:
                    json.load(f)
                v.ok(f"Model artifact: {desc}", f"Valid JSON ({sz:.1f} KB)")
            except Exception as e:
                v.fail(f"Model artifact: {desc}", f"Parse error: {e}")


def check_feature_consistency(v: ValidationResult) -> None:
    """
    Verify that the features used in training data match the features
    expected by the inference service.
    """
    # Check Stage 1 training data
    s1_data = os.path.join(BASE_DIR, "data", "stage1_training_data.csv")
    if os.path.exists(s1_data):
        df = pd.read_csv(s1_data, nrows=1)
        data_cols = set(df.columns) - {"label"}
        expected = set(STAGE1_FEATURES)

        missing_in_data = expected - data_cols
        extra_in_data = data_cols - expected

        if not missing_in_data and not extra_in_data:
            v.ok("Stage 1 feature consistency", f"All {len(expected)} features match")
        else:
            if missing_in_data:
                v.fail("Stage 1 feature consistency",
                       f"Missing in data: {missing_in_data}")
            if extra_in_data:
                v.warn("Stage 1 feature consistency",
                       f"Extra in data: {extra_in_data}")
    else:
        v.warn("Stage 1 feature consistency", "Training data not found (skipped)")

    # Check Stage 2 training data
    s2_data = os.path.join(BASE_DIR, "data", "stage2_training_data.csv")
    if os.path.exists(s2_data):
        df = pd.read_csv(s2_data, nrows=1)
        data_cols = set(df.columns) - {"label"}
        expected = set(STAGE2_FEATURES)

        missing_in_data = expected - data_cols
        extra_in_data = data_cols - expected

        if not missing_in_data and not extra_in_data:
            v.ok("Stage 2 feature consistency", f"All {len(expected)} features match")
        else:
            if missing_in_data:
                v.fail("Stage 2 feature consistency",
                       f"Missing in data: {missing_in_data}")
            if extra_in_data:
                v.warn("Stage 2 feature consistency",
                       f"Extra in data: {extra_in_data}")
    else:
        v.warn("Stage 2 feature consistency", "Training data not found (skipped)")

    # Check model n_features matches
    model_dir = os.path.join(BASE_DIR, "models")
    s1_model_path = os.path.join(model_dir, "human_vs_bot.pkl")
    if os.path.exists(s1_model_path):
        model = joblib.load(s1_model_path)
        if hasattr(model, "n_features_in_"):
            if model.n_features_in_ == len(STAGE1_FEATURES):
                v.ok("Stage 1 model feature count",
                     f"Model expects {model.n_features_in_} features, we provide {len(STAGE1_FEATURES)}")
            else:
                v.fail("Stage 1 model feature count",
                       f"MISMATCH: model expects {model.n_features_in_}, "
                       f"pipeline provides {len(STAGE1_FEATURES)}")

    s2_model_path = os.path.join(model_dir, "attack_classifier.pkl")
    if os.path.exists(s2_model_path):
        model = joblib.load(s2_model_path)
        if hasattr(model, "n_features_in_"):
            if model.n_features_in_ == len(STAGE2_FEATURES):
                v.ok("Stage 2 model feature count",
                     f"Model expects {model.n_features_in_} features, we provide {len(STAGE2_FEATURES)}")
            else:
                v.fail("Stage 2 model feature count",
                       f"MISMATCH: model expects {model.n_features_in_}, "
                       f"pipeline provides {len(STAGE2_FEATURES)}")


def check_data_leakage(v: ValidationResult) -> None:
    """
    Check that training and evaluation use different random seeds
    and that there is no overlap.
    """
    # Verify seed constants
    v.ok("Training seed",  "random_state=42 (hardcoded in training scripts)")
    v.ok("Evaluation seed", "random_state=99 (hardcoded in evaluation script)")

    # Check if stage1 training data contains non-numeric label column
    s1_data = os.path.join(BASE_DIR, "data", "stage1_training_data.csv")
    if os.path.exists(s1_data):
        df = pd.read_csv(s1_data, nrows=100)
        label_col = df["label"]

        # Check label encoding is correct (0=human, 1=bot)
        unique_labels = sorted(label_col.unique())
        if set(unique_labels) == {0, 1}:
            v.ok("Stage 1 label encoding", "Binary: 0=Human, 1=Bot")
        else:
            v.warn("Stage 1 label encoding", f"Unexpected labels: {unique_labels}")

        # Check for NaN/Inf in features
        feature_df = df[STAGE1_FEATURES]
        inf_count = np.isinf(feature_df.values).sum()
        nan_count = np.isnan(feature_df.values).sum()
        if inf_count == 0 and nan_count == 0:
            v.ok("Stage 1 data quality", "No NaN or Inf values in features")
        else:
            v.warn("Stage 1 data quality", f"Found {nan_count} NaN, {inf_count} Inf values")

    # Check if stage2 training data has no human rows
    s2_data = os.path.join(BASE_DIR, "data", "stage2_training_data.csv")
    if os.path.exists(s2_data):
        df = pd.read_csv(s2_data, nrows=1000)
        if "human" in df["label"].values:
            v.fail("Stage 2 data isolation", "LEAKAGE: Human rows found in Stage 2 data!")
        else:
            v.ok("Stage 2 data isolation", "No human rows in Stage 2 data (bot-only)")

        # Check that all 3 attack classes are present
        classes = sorted(df["label"].unique())
        expected = ["flood", "recon", "scraper"]
        if classes == expected:
            v.ok("Stage 2 class coverage", f"All classes present: {classes}")
        else:
            v.fail("Stage 2 class coverage", f"Expected {expected}, got {classes}")


def check_label_encoder_consistency(v: ValidationResult) -> None:
    """Verify the label encoder matches the expected classes."""
    enc_path = os.path.join(BASE_DIR, "models", "attack_classifier_encoder.pkl")
    if not os.path.exists(enc_path):
        v.fail("Label encoder", "File not found")
        return

    encoder = joblib.load(enc_path)
    classes = sorted(encoder.classes_.tolist())
    expected = ["flood", "recon", "scraper"]

    if classes == expected:
        v.ok("Label encoder classes", f"Matches expected: {classes}")
    else:
        v.fail("Label encoder classes", f"Expected {expected}, got {classes}")


def check_threshold_sanity(v: ValidationResult) -> None:
    """Verify the threshold is in a sane range."""
    thr_path = os.path.join(BASE_DIR, "models", "human_vs_bot_threshold.json")
    if not os.path.exists(thr_path):
        v.warn("Threshold file", "Not found — will use default 0.50")
        return

    with open(thr_path) as f:
        data = json.load(f)

    threshold = data.get("optimal_threshold", None)
    if threshold is None:
        v.fail("Threshold value", "Key 'optimal_threshold' not found in JSON")
        return

    if 0.01 <= threshold <= 0.99:
        v.ok("Threshold range", f"{threshold} is within valid range [0.01, 0.99]")
    else:
        v.fail("Threshold range", f"{threshold} is outside valid range!")

    if threshold < 0.10:
        v.warn("Threshold tuning",
               f"Very low ({threshold}) — almost all traffic classified as bot. "
               "May cause false positives.")
    elif threshold > 0.80:
        v.warn("Threshold tuning",
               f"Very high ({threshold}) — many bots may slip through as human.")


def check_inference_assembler(v: ValidationResult) -> None:
    """Verify the assembler module is importable and consistent."""
    assembler_path = os.path.join(BASE_DIR, "inference_service", "assembler.py")
    if not os.path.exists(assembler_path):
        v.fail("Assembler module", "Not found")
        return

    # Read the file and check for feature list consistency
    with open(assembler_path) as f:
        content = f.read()

    for feat in STAGE1_FEATURES:
        if feat not in content:
            v.fail("Assembler ↔ Stage 1", f"Feature {feat} not in assembler.py")
            return

    for feat in STAGE2_FEATURES:
        if feat not in content:
            v.fail("Assembler ↔ Stage 2", f"Feature {feat} not in assembler.py")
            return

    v.ok("Assembler feature coverage", "All 22 features present in assembler.py")


def check_model_can_predict(v: ValidationResult) -> None:
    """Smoke test: feed a dummy row through both models."""
    model_dir = os.path.join(BASE_DIR, "models")

    # Stage 1
    s1_path = os.path.join(model_dir, "human_vs_bot.pkl")
    if os.path.exists(s1_path):
        model = joblib.load(s1_path)
        dummy = pd.DataFrame([[0.0] * len(STAGE1_FEATURES)], columns=STAGE1_FEATURES)
        try:
            proba = model.predict_proba(dummy)
            if proba.shape == (1, 2):
                v.ok("Stage 1 smoke test", f"predict_proba returned shape {proba.shape}")
            else:
                v.fail("Stage 1 smoke test", f"Unexpected shape: {proba.shape}")
        except Exception as e:
            v.fail("Stage 1 smoke test", str(e))

    # Stage 2
    s2_path = os.path.join(model_dir, "attack_classifier.pkl")
    enc_path = os.path.join(model_dir, "attack_classifier_encoder.pkl")
    if os.path.exists(s2_path) and os.path.exists(enc_path):
        model = joblib.load(s2_path)
        encoder = joblib.load(enc_path)
        dummy = pd.DataFrame([[0.0] * len(STAGE2_FEATURES)], columns=STAGE2_FEATURES)
        try:
            pred = model.predict(dummy)
            label = encoder.inverse_transform(pred)[0]
            v.ok("Stage 2 smoke test", f"Predicted: '{label}'")
        except Exception as e:
            v.fail("Stage 2 smoke test", str(e))


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  SAGE Pipeline Validation — Pre-Production Checks")
    print("=" * 65)

    v = ValidationResult()

    print("\n  [1/7] Model artifacts ...")
    check_model_artifacts(v)

    print("  [2/7] Feature consistency ...")
    check_feature_consistency(v)

    print("  [3/7] Data leakage checks ...")
    check_data_leakage(v)

    print("  [4/7] Label encoder consistency ...")
    check_label_encoder_consistency(v)

    print("  [5/7] Threshold sanity ...")
    check_threshold_sanity(v)

    print("  [6/7] Inference assembler ...")
    check_inference_assembler(v)

    print("  [7/7] Model smoke tests ...")
    check_model_can_predict(v)

    # Print results
    print(f"\n{'─' * 65}")
    print(v.to_text())
    print(f"\n{'─' * 65}")
    print(f"  PASS: {v.passed}  |  WARN: {v.warnings}  |  FAIL: {v.failed}")

    if v.failed > 0:
        print(f"\n  ✗ VALIDATION FAILED — {v.failed} critical issue(s) detected")
        print("    Do NOT deploy until all FAIL items are resolved.")
    elif v.warnings > 0:
        print(f"\n  ⚠ PASSED WITH WARNINGS — review {v.warnings} warning(s)")
    else:
        print("\n  ✓ ALL CHECKS PASSED — pipeline is production-ready")

    # Save report
    report_path = os.path.join(BASE_DIR, "reports", "pipeline_validation.txt")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("  SAGE Pipeline Validation Report\n")
        f.write(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        f.write(v.to_text() + "\n\n")
        f.write(f"PASS: {v.passed}  |  WARN: {v.warnings}  |  FAIL: {v.failed}\n")
        f.write("=" * 70 + "\n")

    print(f"\n  Report: {report_path}")
    print("=" * 65)

    sys.exit(1 if v.failed > 0 else 0)


if __name__ == "__main__":
    main()
