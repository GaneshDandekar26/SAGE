# ml_pipeline/scripts/preprocessing/generate_stage2_features.py
"""
Generate Stage 2 training data: Flood vs Scraper vs Recon (3-class)

This dataset does NOT contain any Human/Benign rows.  Stage 2 only
receives traffic already confirmed as bot by Stage 1.

Strategy:
  - Map CIC-IDS2018 attack-type rows to the 10 operational features
    using domain-informed synthetic generation.
  - DDoS-LOIC-HTTP    → label "flood"
  - Bot               → label "scraper"
  - Infiltration      → label "recon"

Features (10 total):
    SAGE_Request_Velocity, SAGE_Peak_Burst_RPS, SAGE_Velocity_Trend,
    SAGE_Endpoint_Concentration, SAGE_Cart_Ratio, SAGE_Sequential_Traversal,
    SAGE_Sensitive_Endpoint_Ratio,
    SAGE_UA_Entropy, SAGE_Header_Completeness, SAGE_Response_Size_Variance
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

np.random.seed(42)


def resolve_base_dir():
    current = Path(__file__).resolve().parent
    for candidate in [current, *current.parents]:
        if candidate.name == "ml_pipeline" and (candidate / "requirements.txt").exists():
            return str(candidate)
    raise RuntimeError("Could not resolve ml_pipeline base directory.")


BASE_DIR = resolve_base_dir()

# CIC-IDS2018 source files with attack-type mappings
SOURCE_FILES = {
    "DDoS attacks-LOIC-HTTP.csv": "flood",
    "Bot.csv": "scraper",
    "Infilteration.csv": "recon",
}

RAW_COLUMNS = [
    "Tot Fwd Pkts", "Tot Bwd Pkts",
    "Flow IAT Mean", "Flow IAT Std",
    "Flow Pkts/s", "Fwd Pkt Len Std",
    "Label",
]

STAGE2_FEATURES = [
    # Volume & Velocity (3)
    "SAGE_Request_Velocity",
    "SAGE_Peak_Burst_RPS",
    "SAGE_Velocity_Trend",
    # Target Selection (4)
    "SAGE_Endpoint_Concentration",
    "SAGE_Cart_Ratio",
    "SAGE_Sequential_Traversal",
    "SAGE_Sensitive_Endpoint_Ratio",
    # Behavioural Fingerprint (3)
    "SAGE_UA_Entropy",
    "SAGE_Header_Completeness",
    "SAGE_Response_Size_Variance",
]


def _safe_numeric(series):
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


# =====================================================================
# Per-class feature generators
# =====================================================================

def _synthesize_flood(cic_df: pd.DataFrame = None, n: int = None) -> pd.DataFrame:
    """
    Flood attack profile:
      - Extremely high velocity and burst RPS
      - Flat velocity trend (sustained, not accelerating)
      - Very high endpoint concentration (single target)
      - Zero cart ratio
      - Zero sequential traversal (same URL repeated)
      - Zero sensitive endpoint ratio (targeting performance)
      - Single user-agent (no rotation) → low UA entropy
      - Missing browser headers → low header completeness
      - Nearly zero response size variance (same response)
    """
    if n is None:
        n = len(cic_df)

    if cic_df is not None:
        velocity_seed = _safe_numeric(cic_df["Flow Pkts/s"]).clip(1, 100000)
    else:
        velocity_seed = pd.Series(np.random.lognormal(6, 1.5, n))

    data = {
        "SAGE_Request_Velocity": (velocity_seed * 0.5 + np.random.uniform(50, 200, n)).clip(30, 5000).values,
        "SAGE_Peak_Burst_RPS": np.random.uniform(30, 500, n),
        "SAGE_Velocity_Trend": np.random.normal(0, 2, n).clip(-5, 5),  # flat trend
        "SAGE_Endpoint_Concentration": np.random.uniform(0.85, 1.0, n),
        "SAGE_Cart_Ratio": np.zeros(n),
        "SAGE_Sequential_Traversal": np.random.exponential(0.02, n).clip(0, 0.1),
        "SAGE_Sensitive_Endpoint_Ratio": np.random.exponential(0.02, n).clip(0, 0.1),
        "SAGE_UA_Entropy": np.random.exponential(0.05, n).clip(0, 0.3),
        "SAGE_Header_Completeness": np.random.uniform(0.0, 0.25, n),
        "SAGE_Response_Size_Variance": np.random.exponential(10, n).clip(0, 50),
    }
    df = pd.DataFrame(data)
    df["label"] = "flood"
    return df


def _synthesize_scraper(cic_df: pd.DataFrame = None, n: int = None) -> pd.DataFrame:
    """
    Scraper attack profile:
      - Moderate velocity (steady crawl, not overwhelming)
      - Moderate burst RPS
      - Flat or slightly decreasing velocity trend
      - Medium endpoint concentration (spread across data pages)
      - Cart ratio = 0 (never purchases)
      - HIGH sequential traversal (enumerating product IDs)
      - Low sensitive endpoint ratio (targets data, not admin)
      - High UA entropy (rotates user agents to evade)
      - High header completeness (mimics real browser)
      - High response size variance (different product pages)
    """
    if n is None:
        n = len(cic_df)

    data = {
        "SAGE_Request_Velocity": np.random.uniform(10, 80, n),
        "SAGE_Peak_Burst_RPS": np.random.uniform(5, 30, n),
        "SAGE_Velocity_Trend": np.random.normal(-1, 3, n).clip(-10, 5),
        "SAGE_Endpoint_Concentration": np.random.uniform(0.3, 0.7, n),
        "SAGE_Cart_Ratio": np.random.exponential(0.005, n).clip(0, 0.03),
        "SAGE_Sequential_Traversal": np.random.uniform(0.4, 0.95, n),
        "SAGE_Sensitive_Endpoint_Ratio": np.random.exponential(0.02, n).clip(0, 0.08),
        "SAGE_UA_Entropy": np.random.uniform(1.5, 3.5, n),
        "SAGE_Header_Completeness": np.random.uniform(0.7, 1.0, n),
        "SAGE_Response_Size_Variance": np.random.uniform(200, 5000, n),
    }
    df = pd.DataFrame(data)
    df["label"] = "scraper"
    return df


def _synthesize_recon(cic_df: pd.DataFrame = None, n: int = None) -> pd.DataFrame:
    """
    Recon/probing attack profile:
      - Low velocity (slow and stealthy)
      - Low burst RPS (avoids detection)
      - Positive velocity trend (probing then escalating)
      - Low endpoint concentration (explores broadly)
      - Zero cart ratio
      - Low sequential traversal (random probing, not enumeration)
      - HIGH sensitive endpoint ratio (targeting admin/config/debug)
      - Low UA entropy (uses common browser UA, no rotation needed)
      - Medium header completeness (tries to look normal)
      - Medium response size variance (mix of 200/404/403 responses)
    """
    if n is None:
        n = len(cic_df)

    data = {
        "SAGE_Request_Velocity": np.random.uniform(2, 25, n),
        "SAGE_Peak_Burst_RPS": np.random.uniform(1, 10, n),
        "SAGE_Velocity_Trend": np.random.uniform(2, 15, n),  # accelerating
        "SAGE_Endpoint_Concentration": np.random.uniform(0.05, 0.35, n),
        "SAGE_Cart_Ratio": np.zeros(n),
        "SAGE_Sequential_Traversal": np.random.exponential(0.1, n).clip(0, 0.3),
        "SAGE_Sensitive_Endpoint_Ratio": np.random.uniform(0.3, 0.9, n),
        "SAGE_UA_Entropy": np.random.uniform(0.0, 1.0, n),
        "SAGE_Header_Completeness": np.random.uniform(0.4, 0.8, n),
        "SAGE_Response_Size_Variance": np.random.uniform(50, 1000, n),
    }
    df = pd.DataFrame(data)
    df["label"] = "recon"
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate Stage 2 training data (Flood/Scraper/Recon)")
    parser.add_argument("--data-dir", default=os.path.join(BASE_DIR, "data"),
                        help="Directory containing CIC-IDS2018 CSV files")
    parser.add_argument("--output", default=os.path.join(BASE_DIR, "data", "stage2_training_data.csv"),
                        help="Output CSV path")
    parser.add_argument("--max-per-file", type=int, default=25000,
                        help="Max rows per source file")
    parser.add_argument("--target-per-class", type=int, default=20000,
                        help="Target rows per class for balanced dataset")
    args = parser.parse_args()

    print("=" * 60)
    print("SAGE Stage 2 Training Data Generator")
    print("=" * 60)

    all_frames = []

    for filename, attack_label in SOURCE_FILES.items():
        filepath = os.path.join(args.data_dir, filename)

        if os.path.exists(filepath):
            print(f"\n  Loading {filename} → '{attack_label}'...")
            df = pd.read_csv(filepath, usecols=RAW_COLUMNS, low_memory=False)
            df.columns = df.columns.str.strip()
            df["Label"] = df["Label"].astype(str).str.strip()

            # Only keep attack rows (not Benign)
            attack_mask = df["Label"] != "Benign"
            attack_rows = df[attack_mask].copy()

            if len(attack_rows) > args.max_per_file:
                attack_rows = attack_rows.sample(n=args.max_per_file, random_state=42)

            print(f"    CIC attack rows: {len(attack_rows)}")

            # Generate synthetic features based on attack type
            if attack_label == "flood":
                synth = _synthesize_flood(cic_df=attack_rows)
            elif attack_label == "scraper":
                synth = _synthesize_scraper(cic_df=attack_rows)
            else:
                synth = _synthesize_recon(cic_df=attack_rows)

            all_frames.append(synth)
        else:
            print(f"\n  [SKIP] {filename} not found, generating purely synthetic data")

    # Fill any missing attack types with purely synthetic data
    existing_labels = set()
    for frame in all_frames:
        existing_labels.update(frame["label"].unique())

    for attack_type in ["flood", "scraper", "recon"]:
        if attack_type not in existing_labels:
            print(f"\n  Generating {args.target_per_class} synthetic '{attack_type}' samples...")
            if attack_type == "flood":
                all_frames.append(_synthesize_flood(n=args.target_per_class))
            elif attack_type == "scraper":
                all_frames.append(_synthesize_scraper(n=args.target_per_class))
            else:
                all_frames.append(_synthesize_recon(n=args.target_per_class))

    # Combine all frames
    combined = pd.concat(all_frames, ignore_index=True)

    # Balance classes
    print(f"\n  Balancing to {args.target_per_class} per class...")
    balanced_frames = []
    for label in ["flood", "scraper", "recon"]:
        subset = combined[combined["label"] == label]
        if len(subset) > args.target_per_class:
            subset = subset.sample(n=args.target_per_class, random_state=42)
        elif len(subset) < args.target_per_class:
            # Oversample by repeating + adding noise
            shortfall = args.target_per_class - len(subset)
            oversampled = subset.sample(n=shortfall, replace=True, random_state=42)
            # Add small noise to avoid exact duplicates
            numeric_cols = [c for c in STAGE2_FEATURES if c in oversampled.columns]
            for col in numeric_cols:
                oversampled[col] = oversampled[col] * np.random.uniform(0.95, 1.05, len(oversampled))
            subset = pd.concat([subset, oversampled], ignore_index=True)
        balanced_frames.append(subset)

    final = pd.concat(balanced_frames, ignore_index=True)
    final = final.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Ensure column order
    final = final[STAGE2_FEATURES + ["label"]]
    final = final.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    final.to_csv(args.output, index=False)

    print(f"\n{'=' * 60}")
    print(f"Stage 2 training data saved: {args.output}")
    print(f"Total rows: {len(final)}")
    print(f"Class distribution:\n{final['label'].value_counts().to_string()}")
    print(f"Features: {STAGE2_FEATURES}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
