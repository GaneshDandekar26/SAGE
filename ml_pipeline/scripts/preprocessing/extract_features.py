import re
import pandas as pd
import numpy as np
from datetime import datetime
import os
from pathlib import Path


def resolve_base_dir():
    current = Path(__file__).resolve().parent
    for candidate in [current, *current.parents]:
        if candidate.name == "ml_pipeline" and (candidate / "requirements.txt").exists():
            return str(candidate)
    raise RuntimeError("Could not resolve ml_pipeline base directory.")


BASE_DIR = resolve_base_dir()

LOG_FILES = [
    {"path": os.path.join(BASE_DIR, "data", "human_logs.txt"), "label": 0},
    {"path": os.path.join(BASE_DIR, "data", "bot_logs.txt"), "label": 1}
]
OUTPUT_CSV = os.path.join(BASE_DIR, "data", "sage_training_data.csv")

LOG_PATTERN = re.compile(
    r'\[(?P<timestamp>.*?)\] '          # 1. Timestamp
    r'"(?P<method>\S+)\s+'              # 2. HTTP Method (GET/POST)
    r'(?P<endpoint>\S+).*?" '           # 3. Requested URL/Endpoint
    r'\d+ \d+ '                         # 4. Status and Bytes (ignored for features)
    r'".*?" '                           # 5. Referrer (ignored)
    r'(?P<session_id>\S+) '             # 6. Session ID (CRITICAL)
    r'".*?"'                            # 7. User Agent (ignored)
)

def parse_logs():
    parsed_data = []

    for file_info in LOG_FILES:
        filepath = file_info["path"]
        label = file_info["label"]

        if not os.path.exists(filepath):
            print(f"Warning: Could not find {filepath}. Skipping.")
            continue

        with open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                match = LOG_PATTERN.search(line)
                if match:
                    data = match.groupdict()
                    # Skip lines that don't have a valid session ID (logged as '-')
                    if data['session_id'] == '-':
                        continue

                    # Convert timestamp string to actual Python datetime object
                    # Format: 01/Aug/2020:12:25:34 +0000
                    dt_obj = datetime.strptime(data['timestamp'], '%d/%b/%Y:%H:%M:%S %z')

                    parsed_data.append({
                        'session_id': data['session_id'],
                        'timestamp': dt_obj,
                        'endpoint': data['endpoint'],
                        'label': label
                    })

    return pd.DataFrame(parsed_data)


def extract_sage_features(df):
    print(f"Processing {len(df)} raw log entries...")

    # Sort chronologically so our time-deltas are accurate
    df = df.sort_values(by=['session_id', 'timestamp'])

    features = []

    # Group all requests by the user's Session ID
    for session_id, group in df.groupby('session_id'):

        #   Session Depth: Total number of requests in this session
        session_depth = len(group)

        # Endpoint Diversity: Ratio of unique endpoints to total requests

        unique_endpoints = group['endpoint'].nunique()
        endpoint_diversity = unique_endpoints / session_depth

        # Temporal Variance & Velocity
        if session_depth > 1:
            # Calculate time difference between each click (in seconds)
            time_deltas = group['timestamp'].diff().dt.total_seconds().dropna()

            temporal_variance = time_deltas.std()
            if pd.isna(temporal_variance):
                temporal_variance = 0.0

            # Total duration of the session in seconds
            duration_seconds = (group['timestamp'].iloc[-1] - group['timestamp'].iloc[0]).total_seconds()

            # Request Velocity (Requests per minute).
            if duration_seconds > 0:
                request_velocity = session_depth / (duration_seconds / 60)
            else:
                request_velocity = session_depth * 60 # All requests happened in same second

        else:
            # Fallbacks for single-request sessions
            temporal_variance = 0.0
            request_velocity = 0.0

        features.append({
            'session_id': session_id,
            'endpoint_diversity': round(endpoint_diversity, 4),
            'temporal_variance': round(temporal_variance, 4),
            'session_depth': session_depth,
            'request_velocity': round(request_velocity, 2),
            'label': group['label'].iloc[0] # 0 for human, 1 for bot
        })

    return pd.DataFrame(features)


if __name__ == "__main__":
    print("Starting SAGE Feature Extraction...")


    raw_df = parse_logs()

    if raw_df.empty:
        print("Error: No data parsed. Check your file paths and log formats.")
    else:

        sage_df = extract_sage_features(raw_df)

        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
        sage_df.to_csv(OUTPUT_CSV, index=False)

        print("\nExtraction Complete! SAGE Gateway Features:")
        print(sage_df.head(10).to_string())
        print(f"\nSaved clean training dataset to {OUTPUT_CSV}")
