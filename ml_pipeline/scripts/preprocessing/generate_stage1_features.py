"""
SAGE Stage 1 — Training Data: Human vs Bot (v2 — Realistic)
=============================================================

Fixes from v1 audit:
  1. Overlapping distributions — human and bot feature ranges overlap by ~30%
  2. Borderline samples       — 15% of each class sits near decision boundary
  3. Session-based IDs        — every row has a session_id for proper splitting
  4. Timestamp ordering       — temporal structure for time-based splits
  5. No shared generator      — train data uses different params than eval

Each "session" represents a user browsing session (10-80 requests).
Features are per-session aggregates, just like real inference.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

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


# ── Distribution helpers ─────────────────────────────────────────────

def _noisy(center: float, spread: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Normal distribution clipped to [0, ∞), with controlled spread."""
    return np.abs(rng.normal(center, spread, n))


def _bounded(low: float, high: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform with Gaussian noise at edges for soft boundaries."""
    base = rng.uniform(low, high, n)
    noise = rng.normal(0, (high - low) * 0.08, n)  # 8% edge bleed
    return np.clip(base + noise, low * 0.7, high * 1.3)


# ── Session generators ───────────────────────────────────────────────

def generate_humans(n: int, rng: np.random.Generator, borderline_frac: float = 0.15) -> pd.DataFrame:
    """
    Generate human session features with realistic distributions.

    ~85% typical humans, ~15% "borderline" humans who browse fast/mechanically
    (these overlap with bot feature ranges).
    """
    n_typical = int(n * (1 - borderline_frac))
    n_border = n - n_typical

    # ── Typical humans: irregular timing, diverse browsing ─────────
    typical = {
        "SAGE_InterArrival_CV":      _noisy(1.4, 0.55, n_typical, rng),      # high variance [~0.3 - 2.5]
        "SAGE_Timing_Entropy":       _bounded(1.0, 2.3, n_typical, rng),      # high entropy
        "SAGE_Pause_Ratio":          _bounded(0.15, 0.65, n_typical, rng),    # lots of pauses
        "SAGE_Burst_Score":          rng.exponential(0.03, n_typical).clip(0, 0.25),  # low bursts [0-0.25]
        "SAGE_Backtrack_Ratio":      _bounded(0.10, 0.55, n_typical, rng),    # frequent backtracking
        "SAGE_Path_Entropy":         _bounded(1.5, 4.0, n_typical, rng),      # diverse pages
        "SAGE_Referral_Chain_Depth": _bounded(1.5, 3.8, n_typical, rng),      # deep navigation
        "SAGE_Session_Depth":        rng.lognormal(2.5, 0.8, n_typical).clip(3, 200).astype(int),
        "SAGE_Method_Diversity":     _bounded(0.08, 0.40, n_typical, rng),    # GET + POST
        "SAGE_Static_Asset_Ratio":   _bounded(0.15, 0.55, n_typical, rng),    # loads CSS/JS
        "SAGE_Error_Rate":           rng.exponential(0.025, n_typical).clip(0, 0.20),
        "SAGE_Payload_Variance":     _bounded(40, 1500, n_typical, rng),
    }

    # ── Borderline humans: fast browsers, power users, API devs ───
    # These intentionally OVERLAP with bot ranges
    border = {
        "SAGE_InterArrival_CV":      _noisy(0.45, 0.20, n_border, rng),       # OVERLAPS with bots
        "SAGE_Timing_Entropy":       _bounded(0.6, 1.5, n_border, rng),       # OVERLAPS
        "SAGE_Pause_Ratio":          _bounded(0.05, 0.25, n_border, rng),     # OVERLAPS
        "SAGE_Burst_Score":          _bounded(0.05, 0.20, n_border, rng),     # OVERLAPS
        "SAGE_Backtrack_Ratio":      _bounded(0.05, 0.25, n_border, rng),     # OVERLAPS
        "SAGE_Path_Entropy":         _bounded(1.0, 2.5, n_border, rng),       # OVERLAPS
        "SAGE_Referral_Chain_Depth": _bounded(1.2, 2.2, n_border, rng),       # OVERLAPS
        "SAGE_Session_Depth":        rng.lognormal(3.0, 1.0, n_border).clip(5, 500).astype(int),
        "SAGE_Method_Diversity":     _bounded(0.05, 0.20, n_border, rng),     # OVERLAPS
        "SAGE_Static_Asset_Ratio":   _bounded(0.05, 0.25, n_border, rng),     # OVERLAPS
        "SAGE_Error_Rate":           _bounded(0.03, 0.18, n_border, rng),     # OVERLAPS
        "SAGE_Payload_Variance":     _bounded(10, 300, n_border, rng),        # OVERLAPS
    }

    df_t = pd.DataFrame(typical)
    df_b = pd.DataFrame(border)
    df = pd.concat([df_t, df_b], ignore_index=True)
    df["label"] = 0
    df["session_type"] = ["typical"] * n_typical + ["borderline"] * n_border
    return df


def generate_bots(n: int, rng: np.random.Generator, borderline_frac: float = 0.15) -> pd.DataFrame:
    """
    Generate bot session features with realistic distributions.

    ~85% obvious bots, ~15% sophisticated bots who mimic human behavior
    (these overlap with human feature ranges).
    """
    n_obvious = int(n * (1 - borderline_frac))
    n_stealth = n - n_obvious

    # ── Obvious bots: metronomic timing, no browsing variety ──────
    obvious = {
        "SAGE_InterArrival_CV":      _noisy(0.15, 0.12, n_obvious, rng),       # low variance
        "SAGE_Timing_Entropy":       _bounded(0.0, 1.0, n_obvious, rng),       # low entropy
        "SAGE_Pause_Ratio":          rng.exponential(0.03, n_obvious).clip(0, 0.15),  # no reading time
        "SAGE_Burst_Score":          _bounded(0.10, 0.55, n_obvious, rng),     # rapid fire
        "SAGE_Backtrack_Ratio":      rng.exponential(0.04, n_obvious).clip(0, 0.15),  # no backtracking
        "SAGE_Path_Entropy":         _bounded(0.1, 1.8, n_obvious, rng),       # repetitive
        "SAGE_Referral_Chain_Depth": _bounded(1.0, 1.7, n_obvious, rng),       # shallow navigation
        "SAGE_Session_Depth":        rng.lognormal(3.5, 1.0, n_obvious).clip(5, 5000).astype(int),
        "SAGE_Method_Diversity":     rng.exponential(0.025, n_obvious).clip(0, 0.12),  # GET only
        "SAGE_Static_Asset_Ratio":   rng.exponential(0.02, n_obvious).clip(0, 0.10),   # no assets
        "SAGE_Error_Rate":           _bounded(0.05, 0.45, n_obvious, rng),     # high errors
        "SAGE_Payload_Variance":     _bounded(0, 80, n_obvious, rng),          # uniform payloads
    }

    # ── Stealth bots: scrapers/recon that mimic humans ────────────
    # These intentionally OVERLAP with human ranges
    stealth = {
        "SAGE_InterArrival_CV":      _noisy(0.70, 0.30, n_stealth, rng),       # OVERLAPS
        "SAGE_Timing_Entropy":       _bounded(0.8, 1.8, n_stealth, rng),       # OVERLAPS
        "SAGE_Pause_Ratio":          _bounded(0.10, 0.35, n_stealth, rng),     # OVERLAPS
        "SAGE_Burst_Score":          _bounded(0.02, 0.15, n_stealth, rng),     # OVERLAPS
        "SAGE_Backtrack_Ratio":      _bounded(0.05, 0.20, n_stealth, rng),     # OVERLAPS
        "SAGE_Path_Entropy":         _bounded(1.2, 2.8, n_stealth, rng),       # OVERLAPS
        "SAGE_Referral_Chain_Depth": _bounded(1.3, 2.5, n_stealth, rng),       # OVERLAPS
        "SAGE_Session_Depth":        rng.lognormal(3.0, 0.8, n_stealth).clip(5, 1000).astype(int),
        "SAGE_Method_Diversity":     _bounded(0.05, 0.20, n_stealth, rng),     # OVERLAPS
        "SAGE_Static_Asset_Ratio":   _bounded(0.05, 0.30, n_stealth, rng),     # OVERLAPS (loads some assets)
        "SAGE_Error_Rate":           _bounded(0.02, 0.15, n_stealth, rng),     # OVERLAPS (careful probing)
        "SAGE_Payload_Variance":     _bounded(20, 400, n_stealth, rng),        # OVERLAPS
    }

    df_o = pd.DataFrame(obvious)
    df_s = pd.DataFrame(stealth)
    df = pd.concat([df_o, df_s], ignore_index=True)
    df["label"] = 1
    df["session_type"] = ["obvious"] * n_obvious + ["stealth"] * n_stealth
    return df


def add_session_metadata(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Add session_id and timestamp for proper splitting.
    Sessions are assigned chronologically — earlier sessions go to train,
    later sessions go to test.
    """
    n = len(df)

    # Assign session IDs
    df["session_id"] = [f"sess_{i:06d}" for i in range(n)]

    # Assign timestamps: monotonically increasing with some jitter
    # Simulates 7 days of traffic (in seconds)
    base_ts = 1700000000  # arbitrary epoch
    total_seconds = 7 * 24 * 3600
    df["timestamp"] = base_ts + np.sort(
        rng.uniform(0, total_seconds, n)
    ).astype(int)

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Generate Stage 1 training data (v2 — realistic)"
    )
    parser.add_argument("--output",
                        default=os.path.join(BASE_DIR, "data", "stage1_training_data.csv"))
    parser.add_argument("--n-human", type=int, default=50000)
    parser.add_argument("--n-bot", type=int, default=50000)
    parser.add_argument("--borderline-frac", type=float, default=0.15,
                        help="Fraction of borderline/hard samples per class")
    args = parser.parse_args()

    print("=" * 65)
    print("  SAGE Stage 1 Training Data Generator (v2 — Realistic)")
    print("=" * 65)

    rng = np.random.default_rng(RANDOM_STATE)

    # Generate sessions
    print(f"\n[1] Generating {args.n_human:,} human sessions "
          f"({args.borderline_frac:.0%} borderline)...")
    humans = generate_humans(args.n_human, rng, args.borderline_frac)

    print(f"[2] Generating {args.n_bot:,} bot sessions "
          f"({args.borderline_frac:.0%} stealth)...")
    bots = generate_bots(args.n_bot, rng, args.borderline_frac)

    # Combine and add temporal structure
    print("[3] Adding session IDs and timestamps...")
    combined = pd.concat([humans, bots], ignore_index=True)
    combined = combined.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    combined = add_session_metadata(combined, rng)

    # Clean up
    combined = combined.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    # Save
    output_cols = ["session_id", "timestamp"] + STAGE1_FEATURES + ["label", "session_type"]
    combined = combined[output_cols]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    combined.to_csv(args.output, index=False)

    # Stats
    print(f"\n{'─' * 65}")
    print(f"  Output:      {args.output}")
    print(f"  Total rows:  {len(combined):,}")
    print(f"  Classes:     {combined['label'].value_counts().to_dict()}")
    print(f"  Session types:")
    for st, count in combined["session_type"].value_counts().items():
        print(f"    {st}: {count:,}")

    # Show distribution overlap evidence
    print(f"\n  DISTRIBUTION OVERLAP CHECK:")
    for feat in STAGE1_FEATURES:
        h = combined[combined["label"] == 0][feat]
        b = combined[combined["label"] == 1][feat]
        overlap_low = max(h.min(), b.min())
        overlap_high = min(h.max(), b.max())
        if overlap_high > overlap_low:
            overlap_range = overlap_high - overlap_low
            total_range = max(h.max(), b.max()) - min(h.min(), b.min())
            pct = overlap_range / total_range * 100
            print(f"    {feat:<35s} overlap: {pct:5.1f}%  [{overlap_low:.3f} — {overlap_high:.3f}]")
        else:
            print(f"    {feat:<35s} overlap: NONE ← PROBLEM (gap exists)")

    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
