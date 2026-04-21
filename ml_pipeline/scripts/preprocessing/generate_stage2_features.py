"""
SAGE Stage 2 — Training Data: Flood / Scraper / Recon (v2 — Realistic)
=======================================================================

Fixes from v1:
  1. Overlapping distributions — attack types share feature ranges (~25%)
  2. Borderline samples       — 15% ambiguous sessions per class
  3. Session IDs              — for proper split validation
  4. Timestamps               — for temporal split
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

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


def _noisy(center: float, spread: float, n: int, rng) -> np.ndarray:
    return np.abs(rng.normal(center, spread, n))


def _bounded(low: float, high: float, n: int, rng) -> np.ndarray:
    base = rng.uniform(low, high, n)
    noise = rng.normal(0, (high - low) * 0.10, n)
    return np.clip(base + noise, max(0, low * 0.6), high * 1.4)


# ── Per-class generators (with overlap) ──────────────────────────────

def _gen_flood(n: int, rng, borderline_frac: float = 0.15) -> pd.DataFrame:
    n_core = int(n * (1 - borderline_frac))
    n_edge = n - n_core

    # Core floods: extreme velocity, single target, no rotation
    core = {
        "SAGE_Request_Velocity":       _bounded(80, 500, n_core, rng),
        "SAGE_Peak_Burst_RPS":         _bounded(50, 500, n_core, rng),
        "SAGE_Velocity_Trend":         _noisy(0, 3, n_core, rng) * rng.choice([-1, 1], n_core),
        "SAGE_Endpoint_Concentration": _bounded(0.75, 1.0, n_core, rng),
        "SAGE_Cart_Ratio":             rng.exponential(0.005, n_core).clip(0, 0.03),
        "SAGE_Sequential_Traversal":   rng.exponential(0.03, n_core).clip(0, 0.15),
        "SAGE_Sensitive_Endpoint_Ratio": rng.exponential(0.03, n_core).clip(0, 0.12),
        "SAGE_UA_Entropy":             rng.exponential(0.08, n_core).clip(0, 0.5),
        "SAGE_Header_Completeness":    _bounded(0.0, 0.35, n_core, rng),
        "SAGE_Response_Size_Variance": rng.exponential(15, n_core).clip(0, 80),
    }

    # Borderline floods: moderate velocity (overlaps with scrapers)
    edge = {
        "SAGE_Request_Velocity":       _bounded(25, 120, n_edge, rng),       # OVERLAPS scraper
        "SAGE_Peak_Burst_RPS":         _bounded(15, 80, n_edge, rng),        # OVERLAPS
        "SAGE_Velocity_Trend":         _noisy(0, 4, n_edge, rng) * rng.choice([-1, 1], n_edge),
        "SAGE_Endpoint_Concentration": _bounded(0.55, 0.85, n_edge, rng),    # OVERLAPS
        "SAGE_Cart_Ratio":             rng.exponential(0.008, n_edge).clip(0, 0.05),
        "SAGE_Sequential_Traversal":   _bounded(0.05, 0.25, n_edge, rng),    # OVERLAPS
        "SAGE_Sensitive_Endpoint_Ratio": _bounded(0.03, 0.20, n_edge, rng),  # OVERLAPS recon
        "SAGE_UA_Entropy":             _bounded(0.1, 0.8, n_edge, rng),      # OVERLAPS
        "SAGE_Header_Completeness":    _bounded(0.15, 0.55, n_edge, rng),    # OVERLAPS
        "SAGE_Response_Size_Variance": _bounded(20, 200, n_edge, rng),       # OVERLAPS
    }

    df = pd.concat([pd.DataFrame(core), pd.DataFrame(edge)], ignore_index=True)
    df["label"] = "flood"
    return df


def _gen_scraper(n: int, rng, borderline_frac: float = 0.15) -> pd.DataFrame:
    n_core = int(n * (1 - borderline_frac))
    n_edge = n - n_core

    # Core scrapers: moderate velocity, high sequential traversal, UA rotation
    core = {
        "SAGE_Request_Velocity":       _bounded(10, 80, n_core, rng),
        "SAGE_Peak_Burst_RPS":         _bounded(5, 35, n_core, rng),
        "SAGE_Velocity_Trend":         _noisy(-1, 3, n_core, rng) * rng.choice([-1, 1], n_core),
        "SAGE_Endpoint_Concentration": _bounded(0.25, 0.65, n_core, rng),
        "SAGE_Cart_Ratio":             rng.exponential(0.005, n_core).clip(0, 0.04),
        "SAGE_Sequential_Traversal":   _bounded(0.40, 0.92, n_core, rng),
        "SAGE_Sensitive_Endpoint_Ratio": rng.exponential(0.03, n_core).clip(0, 0.10),
        "SAGE_UA_Entropy":             _bounded(1.2, 3.2, n_core, rng),
        "SAGE_Header_Completeness":    _bounded(0.65, 1.0, n_core, rng),
        "SAGE_Response_Size_Variance": _bounded(150, 4000, n_core, rng),
    }

    # Borderline scrapers: look like recon or flood
    edge = {
        "SAGE_Request_Velocity":       _bounded(5, 30, n_edge, rng),          # OVERLAPS recon
        "SAGE_Peak_Burst_RPS":         _bounded(2, 15, n_edge, rng),          # OVERLAPS
        "SAGE_Velocity_Trend":         _bounded(1, 8, n_edge, rng),           # OVERLAPS recon
        "SAGE_Endpoint_Concentration": _bounded(0.15, 0.45, n_edge, rng),     # OVERLAPS recon
        "SAGE_Cart_Ratio":             rng.exponential(0.003, n_edge).clip(0, 0.02),
        "SAGE_Sequential_Traversal":   _bounded(0.15, 0.50, n_edge, rng),     # OVERLAPS
        "SAGE_Sensitive_Endpoint_Ratio": _bounded(0.10, 0.40, n_edge, rng),   # OVERLAPS recon
        "SAGE_UA_Entropy":             _bounded(0.5, 1.5, n_edge, rng),       # OVERLAPS flood
        "SAGE_Header_Completeness":    _bounded(0.40, 0.75, n_edge, rng),     # OVERLAPS
        "SAGE_Response_Size_Variance": _bounded(50, 500, n_edge, rng),        # OVERLAPS
    }

    df = pd.concat([pd.DataFrame(core), pd.DataFrame(edge)], ignore_index=True)
    df["label"] = "scraper"
    return df


def _gen_recon(n: int, rng, borderline_frac: float = 0.15) -> pd.DataFrame:
    n_core = int(n * (1 - borderline_frac))
    n_edge = n - n_core

    # Core recon: low velocity, high sensitive ratio, accelerating
    core = {
        "SAGE_Request_Velocity":       _bounded(2, 25, n_core, rng),
        "SAGE_Peak_Burst_RPS":         _bounded(1, 12, n_core, rng),
        "SAGE_Velocity_Trend":         _bounded(2, 15, n_core, rng),
        "SAGE_Endpoint_Concentration": _bounded(0.05, 0.35, n_core, rng),
        "SAGE_Cart_Ratio":             rng.exponential(0.002, n_core).clip(0, 0.02),
        "SAGE_Sequential_Traversal":   rng.exponential(0.08, n_core).clip(0, 0.30),
        "SAGE_Sensitive_Endpoint_Ratio": _bounded(0.30, 0.88, n_core, rng),
        "SAGE_UA_Entropy":             _bounded(0.0, 1.0, n_core, rng),
        "SAGE_Header_Completeness":    _bounded(0.35, 0.75, n_core, rng),
        "SAGE_Response_Size_Variance": _bounded(40, 900, n_core, rng),
    }

    # Borderline recon: looks like scraper (higher velocity, some traversal)
    edge = {
        "SAGE_Request_Velocity":       _bounded(15, 60, n_edge, rng),          # OVERLAPS scraper
        "SAGE_Peak_Burst_RPS":         _bounded(8, 30, n_edge, rng),           # OVERLAPS
        "SAGE_Velocity_Trend":         _bounded(-2, 6, n_edge, rng),           # OVERLAPS scraper
        "SAGE_Endpoint_Concentration": _bounded(0.20, 0.55, n_edge, rng),      # OVERLAPS
        "SAGE_Cart_Ratio":             rng.exponential(0.004, n_edge).clip(0, 0.03),
        "SAGE_Sequential_Traversal":   _bounded(0.15, 0.50, n_edge, rng),      # OVERLAPS scraper
        "SAGE_Sensitive_Endpoint_Ratio": _bounded(0.12, 0.45, n_edge, rng),    # OVERLAPS
        "SAGE_UA_Entropy":             _bounded(0.5, 1.8, n_edge, rng),        # OVERLAPS
        "SAGE_Header_Completeness":    _bounded(0.50, 0.85, n_edge, rng),      # OVERLAPS
        "SAGE_Response_Size_Variance": _bounded(100, 2000, n_edge, rng),       # OVERLAPS
    }

    df = pd.concat([pd.DataFrame(core), pd.DataFrame(edge)], ignore_index=True)
    df["label"] = "recon"
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Generate Stage 2 training data (v2 — realistic)"
    )
    parser.add_argument("--output",
                        default=os.path.join(BASE_DIR, "data", "stage2_training_data.csv"))
    parser.add_argument("--n-per-class", type=int, default=20000)
    parser.add_argument("--borderline-frac", type=float, default=0.15)
    args = parser.parse_args()

    print("=" * 65)
    print("  SAGE Stage 2 Training Data Generator (v2 — Realistic)")
    print("=" * 65)

    rng = np.random.default_rng(RANDOM_STATE)

    frames = []
    for name, gen_fn in [("flood", _gen_flood), ("scraper", _gen_scraper), ("recon", _gen_recon)]:
        print(f"  Generating {args.n_per_class:,} '{name}' sessions...")
        df = gen_fn(args.n_per_class, rng, args.borderline_frac)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    # Add session metadata
    n = len(combined)
    combined["session_id"] = [f"bot_sess_{i:06d}" for i in range(n)]
    base_ts = 1700000000
    combined["timestamp"] = base_ts + np.sort(
        rng.uniform(0, 7 * 24 * 3600, n)
    ).astype(int)

    combined = combined.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    output_cols = ["session_id", "timestamp"] + STAGE2_FEATURES + ["label"]
    combined = combined[output_cols]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    combined.to_csv(args.output, index=False)

    print(f"\n  Output:  {args.output}")
    print(f"  Total:   {len(combined):,}")
    print(f"  Classes: {combined['label'].value_counts().to_dict()}")
    print("=" * 65)


if __name__ == "__main__":
    main()
