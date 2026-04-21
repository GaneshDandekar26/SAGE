# ml_pipeline/scripts/preprocessing/generate_stage1_features.py
"""
Generate Stage 1 training data: Human vs Bot (binary)

Strategy:
  - Map CIC-IDS2018 network-level features to our 12 L7 behavioural features
    using domain-informed synthetic generation.
  - "Benign" rows → label 0 (human)
  - All attack rows (Bot, DDoS, Infiltration) → label 1 (bot)
  - Output: a balanced CSV ready for XGBoost training.

The CIC-IDS2018 columns we use as *seeds* for synthetic generation:
    Tot Fwd Pkts, Tot Bwd Pkts       → session depth proxy
    Flow IAT Mean, Flow IAT Std      → timing pattern seed
    Flow Pkts/s                      → velocity seed
    Fwd Pkt Len Std                  → payload diversity seed
    Label                            → ground truth
"""

import argparse
import os
import sys
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

# CIC-IDS2018 source files with label mappings
SOURCE_FILES = {
    "Bot.csv": {"Benign": 0, "Bot": 1},
    "DDoS attacks-LOIC-HTTP.csv": {"Benign": 0, "DDoS attacks-LOIC-HTTP": 1},
    "Infilteration.csv": {"Benign": 0, "Infilteration": 1, "Infiltration": 1},
    "Brute Force -Web.csv": {"Benign": 0, "Brute Force -Web": 1},
}

RAW_COLUMNS = [
    "Tot Fwd Pkts", "Tot Bwd Pkts",
    "Flow IAT Mean", "Flow IAT Std",
    "Flow Pkts/s", "Fwd Pkt Len Std",
    "Label",
]

STAGE1_FEATURES = [
    # Temporal Micro-Patterns (4)
    "SAGE_InterArrival_CV",
    "SAGE_Timing_Entropy",
    "SAGE_Pause_Ratio",
    "SAGE_Burst_Score",
    # Navigation Patterns (4)
    "SAGE_Backtrack_Ratio",
    "SAGE_Path_Entropy",
    "SAGE_Referral_Chain_Depth",
    "SAGE_Session_Depth",
    # Content Interaction (4)
    "SAGE_Method_Diversity",
    "SAGE_Static_Asset_Ratio",
    "SAGE_Error_Rate",
    "SAGE_Payload_Variance",
]


def _safe_numeric(series):
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def _synthesize_human_features(n: int) -> pd.DataFrame:
    """
    Generate realistic human session feature vectors.

    Human characteristics:
      - High timing variance (irregular clicks)
      - Significant pause ratio (reading time > 2s)
      - Low burst score (rarely rapid-fire)
      - High backtrack ratio (re-visits pages)
      - High path entropy (explores diverse pages)
      - Moderate session depth
      - Uses multiple HTTP methods (GET + POST)
      - Loads static assets (.css, .js, images)
      - Low error rate
      - Varied payload sizes (form fills, searches)
    """
    data = {
        "SAGE_InterArrival_CV": np.random.uniform(0.5, 2.5, n),
        "SAGE_Timing_Entropy": np.random.uniform(1.5, 2.3, n),  # max log2(5) ≈ 2.32
        "SAGE_Pause_Ratio": np.random.uniform(0.2, 0.7, n),
        "SAGE_Burst_Score": np.random.exponential(0.01, n).clip(0, 0.1),
        "SAGE_Backtrack_Ratio": np.random.uniform(0.15, 0.6, n),
        "SAGE_Path_Entropy": np.random.uniform(2.0, 4.5, n),
        "SAGE_Referral_Chain_Depth": np.random.uniform(1.5, 4.0, n),
        "SAGE_Session_Depth": np.random.lognormal(mean=2.5, sigma=0.8, size=n).clip(3, 200).astype(int),
        "SAGE_Method_Diversity": np.random.uniform(0.1, 0.4, n),
        "SAGE_Static_Asset_Ratio": np.random.uniform(0.2, 0.6, n),
        "SAGE_Error_Rate": np.random.exponential(0.02, n).clip(0, 0.15),
        "SAGE_Payload_Variance": np.random.uniform(50, 2000, n),
    }
    df = pd.DataFrame(data)
    df["label"] = 0
    return df


def _synthesize_bot_from_cic(cic_df: pd.DataFrame) -> pd.DataFrame:
    """
    Map CIC-IDS2018 network features to our L7 behavioural features
    for bot samples.  Uses the network-level signals as seeds and adds
    controlled noise to simulate realistic L7 patterns.

    Bot characteristics:
      - Low timing variance (metronomic or near-zero gaps)
      - Low pause ratio (no reading time)
      - Higher burst score (sustained rapid requests)
      - Low backtrack ratio (forward-only crawl or single-endpoint flood)
      - Variable path entropy depending on bot type
      - Uses mostly GET
      - Skips static assets
      - Higher error rate (probing)
      - Low payload variance (empty or uniform bodies)
    """
    n = len(cic_df)

    # Use CIC-IDS timing as seed for our timing features
    iat_mean = _safe_numeric(cic_df["Flow IAT Mean"])
    iat_std = _safe_numeric(cic_df["Flow IAT Std"])
    velocity = _safe_numeric(cic_df["Flow Pkts/s"])
    pkt_len_std = _safe_numeric(cic_df["Fwd Pkt Len Std"])
    depth = (
        _safe_numeric(cic_df["Tot Fwd Pkts"]) + _safe_numeric(cic_df["Tot Bwd Pkts"])
    )

    # Temporal: bots have LOW CV, LOW entropy, LOW pause, HIGH burst
    cv_seed = (iat_std / (iat_mean + 1e-6)).clip(0, 10)
    data = {
        "SAGE_InterArrival_CV": (cv_seed * 0.15 + np.random.uniform(0, 0.2, n)).clip(0, 0.5),
        "SAGE_Timing_Entropy": np.random.uniform(0.0, 1.2, n),
        "SAGE_Pause_Ratio": np.random.exponential(0.03, n).clip(0, 0.15),
        "SAGE_Burst_Score": np.random.uniform(0.05, 0.5, n),
        # Navigation: bots have LOW backtrack, variable path entropy
        "SAGE_Backtrack_Ratio": np.random.exponential(0.05, n).clip(0, 0.15),
        "SAGE_Path_Entropy": np.random.uniform(0.1, 2.0, n),
        "SAGE_Referral_Chain_Depth": np.random.uniform(1.0, 1.8, n),
        "SAGE_Session_Depth": depth.clip(1, 5000).values,
        # Content: bots use GET only, skip assets
        "SAGE_Method_Diversity": np.random.exponential(0.02, n).clip(0, 0.1),
        "SAGE_Static_Asset_Ratio": np.random.exponential(0.02, n).clip(0, 0.08),
        "SAGE_Error_Rate": np.random.uniform(0.05, 0.4, n),
        "SAGE_Payload_Variance": (pkt_len_std * 0.1 + np.random.uniform(0, 30, n)).clip(0, 100),
    }
    df = pd.DataFrame(data)
    df["label"] = 1
    return df


def load_and_transform(data_dir: str, max_per_file: int = None):
    """Load CIC-IDS2018 files and generate synthetic L7 features."""
    human_frames = []
    bot_frames = []

    for filename, label_map in SOURCE_FILES.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"  [SKIP] {filename} not found")
            continue

        print(f"  Loading {filename}...")
        df = pd.read_csv(filepath, usecols=RAW_COLUMNS, low_memory=False)
        df.columns = df.columns.str.strip()
        df["Label"] = df["Label"].astype(str).str.strip()

        for raw_label, binary_label in label_map.items():
            mask = df["Label"] == raw_label
            subset = df[mask].copy()

            if max_per_file and len(subset) > max_per_file:
                subset = subset.sample(n=max_per_file, random_state=42)

            if binary_label == 0:
                human_frames.append(subset)
            else:
                bot_frames.append(subset)

    return human_frames, bot_frames


def main():
    parser = argparse.ArgumentParser(description="Generate Stage 1 training data (Human vs Bot)")
    parser.add_argument("--data-dir", default=os.path.join(BASE_DIR, "data"),
                        help="Directory containing CIC-IDS2018 CSV files")
    parser.add_argument("--output", default=os.path.join(BASE_DIR, "data", "stage1_training_data.csv"),
                        help="Output CSV path")
    parser.add_argument("--max-per-file", type=int, default=30000,
                        help="Max rows per label per source file")
    parser.add_argument("--synthetic-humans", type=int, default=25000,
                        help="Number of purely synthetic human sessions to generate")
    parser.add_argument("--target-total", type=int, default=100000,
                        help="Target total rows (balanced 50/50)")
    args = parser.parse_args()

    print("=" * 60)
    print("SAGE Stage 1 Training Data Generator")
    print("=" * 60)

    # 1. Load CIC-IDS2018 data
    print("\n[1] Loading CIC-IDS2018 source files...")
    human_frames, bot_frames = load_and_transform(args.data_dir, args.max_per_file)

    # 2. Generate bot features from CIC data
    print("\n[2] Synthesizing bot features from CIC-IDS network data...")
    bot_dfs = []
    for frame in bot_frames:
        bot_dfs.append(_synthesize_bot_from_cic(frame))
    cic_bots = pd.concat(bot_dfs, ignore_index=True) if bot_dfs else pd.DataFrame()
    print(f"    Bot samples from CIC: {len(cic_bots)}")

    # 3. Generate synthetic human data
    # We combine CIC Benign rows (mapped to L7) with purely synthetic humans
    print(f"\n[3] Generating {args.synthetic_humans} synthetic human sessions...")
    synthetic_humans = _synthesize_human_features(args.synthetic_humans)

    # Also transform CIC Benign rows into L7 human features
    human_dfs = [synthetic_humans]
    for frame in human_frames:
        # For benign CIC rows, generate human-like L7 features with slight CIC influence
        n = len(frame)
        depth = (_safe_numeric(frame["Tot Fwd Pkts"]) + _safe_numeric(frame["Tot Bwd Pkts"])).clip(3, 200)
        human_from_cic = _synthesize_human_features(n)
        human_from_cic["SAGE_Session_Depth"] = depth.values.astype(int)
        human_dfs.append(human_from_cic)

    all_humans = pd.concat(human_dfs, ignore_index=True)
    print(f"    Total human samples: {len(all_humans)}")

    # 4. Balance the dataset
    target_per_class = args.target_total // 2
    print(f"\n[4] Balancing to {target_per_class} per class...")

    if len(all_humans) > target_per_class:
        all_humans = all_humans.sample(n=target_per_class, random_state=42)
    if len(cic_bots) > target_per_class:
        cic_bots = cic_bots.sample(n=target_per_class, random_state=42)

    # If we have fewer bots than target, generate more synthetic ones
    if len(cic_bots) < target_per_class:
        shortfall = target_per_class - len(cic_bots)
        print(f"    Generating {shortfall} additional synthetic bot samples...")
        extra_bots = _synthesize_generic_bots(shortfall)
        cic_bots = pd.concat([cic_bots, extra_bots], ignore_index=True)

    # If we have fewer humans than target, generate more
    if len(all_humans) < target_per_class:
        shortfall = target_per_class - len(all_humans)
        extra_humans = _synthesize_human_features(shortfall)
        all_humans = pd.concat([all_humans, extra_humans], ignore_index=True)

    # 5. Combine and shuffle
    final = pd.concat([all_humans, cic_bots], ignore_index=True)
    final = final.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Ensure correct column order
    final = final[STAGE1_FEATURES + ["label"]]

    # Replace any infinities or NaN
    final = final.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    # 6. Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    final.to_csv(args.output, index=False)

    print(f"\n{'=' * 60}")
    print(f"Stage 1 training data saved: {args.output}")
    print(f"Total rows: {len(final)}")
    print(f"Class distribution:\n{final['label'].value_counts().to_string()}")
    print(f"Features: {STAGE1_FEATURES}")
    print(f"{'=' * 60}")


def _synthesize_generic_bots(n: int) -> pd.DataFrame:
    """Generate generic bot samples without CIC seed data."""
    data = {
        "SAGE_InterArrival_CV": np.random.uniform(0.0, 0.4, n),
        "SAGE_Timing_Entropy": np.random.uniform(0.0, 1.0, n),
        "SAGE_Pause_Ratio": np.random.exponential(0.02, n).clip(0, 0.1),
        "SAGE_Burst_Score": np.random.uniform(0.1, 0.6, n),
        "SAGE_Backtrack_Ratio": np.random.exponential(0.03, n).clip(0, 0.1),
        "SAGE_Path_Entropy": np.random.uniform(0.1, 1.5, n),
        "SAGE_Referral_Chain_Depth": np.random.uniform(1.0, 1.5, n),
        "SAGE_Session_Depth": np.random.lognormal(mean=3.5, sigma=1.0, size=n).clip(5, 5000).astype(int),
        "SAGE_Method_Diversity": np.random.exponential(0.015, n).clip(0, 0.08),
        "SAGE_Static_Asset_Ratio": np.random.exponential(0.01, n).clip(0, 0.05),
        "SAGE_Error_Rate": np.random.uniform(0.05, 0.5, n),
        "SAGE_Payload_Variance": np.random.uniform(0, 50, n),
    }
    df = pd.DataFrame(data)
    df["label"] = 1
    return df


if __name__ == "__main__":
    main()
