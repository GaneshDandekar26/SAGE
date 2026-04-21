import redis
import numpy as np
import pandas as pd

# Stage 1 features: behavioural signals for Human vs Bot
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

# Stage 2 features: operational signals for Flood/Scraper/Recon
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

# Union of all features (22 total)
ALL_FEATURES = list(dict.fromkeys(STAGE1_FEATURES + STAGE2_FEATURES))

# Defaults that represent a "new/unknown" session (human-leaning baseline)
_DEFAULTS = {
    "SAGE_InterArrival_CV": 1.0,       # moderate variance (human-like)
    "SAGE_Timing_Entropy": 1.5,        # moderate entropy
    "SAGE_Pause_Ratio": 0.3,           # some pauses
    "SAGE_Burst_Score": 0.01,          # minimal bursts
    "SAGE_Backtrack_Ratio": 0.2,       # some revisits
    "SAGE_Path_Entropy": 2.0,          # moderate diversity
    "SAGE_Referral_Chain_Depth": 2.0,
    "SAGE_Session_Depth": 1.0,
    "SAGE_Method_Diversity": 0.15,
    "SAGE_Static_Asset_Ratio": 0.3,
    "SAGE_Error_Rate": 0.02,
    "SAGE_Payload_Variance": 100.0,
    "SAGE_Request_Velocity": 1.0,
    "SAGE_Peak_Burst_RPS": 1.0,
    "SAGE_Velocity_Trend": 0.0,
    "SAGE_Endpoint_Concentration": 0.3,
    "SAGE_Cart_Ratio": 0.1,
    "SAGE_Sequential_Traversal": 0.05,
    "SAGE_Sensitive_Endpoint_Ratio": 0.01,
    "SAGE_UA_Entropy": 0.0,
    "SAGE_Header_Completeness": 0.9,
    "SAGE_Response_Size_Variance": 200.0,
}


class FeatureAssembler:
    """
    Assembles the full 22-feature vector from Redis telemetry state
    and provides Stage 1 / Stage 2 sliced views for the cascaded models.
    """

    def __init__(self, host="localhost", port=6379, db=0):
        self.redis_pool = redis.ConnectionPool(
            host=host, port=port, db=db, decode_responses=True
        )
        self.r = redis.Redis(connection_pool=self.redis_pool)

    def assemble_full(self, user_ip: str) -> dict:
        """
        Fetch all 22 features for a user from Redis.
        Returns a dict of {feature_name: float_value}.
        """
        redis_key = f"sage:telemetry:{user_ip}"
        raw_data = self.r.hgetall(redis_key)

        features = {}
        for feat_name in ALL_FEATURES:
            raw_val = raw_data.get(feat_name)
            if raw_val is not None:
                try:
                    features[feat_name] = float(raw_val)
                except (ValueError, TypeError):
                    features[feat_name] = _DEFAULTS.get(feat_name, 0.0)
            else:
                features[feat_name] = _DEFAULTS.get(feat_name, 0.0)

        return features

    def assemble_stage1(self, feature_dict: dict) -> pd.DataFrame:
        """
        Extract Stage 1 features (12 behavioural) as a DataFrame row
        ready for the XGBoost binary classifier.
        """
        row = [feature_dict.get(f, _DEFAULTS.get(f, 0.0)) for f in STAGE1_FEATURES]
        return pd.DataFrame([row], columns=STAGE1_FEATURES)

    def assemble_stage2(self, feature_dict: dict) -> pd.DataFrame:
        """
        Extract Stage 2 features (10 operational) as a DataFrame row
        ready for the Random Forest classifier.
        """
        row = [feature_dict.get(f, _DEFAULTS.get(f, 0.0)) for f in STAGE2_FEATURES]
        return pd.DataFrame([row], columns=STAGE2_FEATURES)

    def assemble_from_payload(self, payload: dict) -> dict:
        """
        Build a feature dict from a raw request payload (from the Java gateway).
        Merges payload values with defaults for any missing features.
        """
        features = {}
        for feat_name in ALL_FEATURES:
            val = payload.get(feat_name)
            if val is not None:
                try:
                    features[feat_name] = float(val)
                except (ValueError, TypeError):
                    features[feat_name] = _DEFAULTS.get(feat_name, 0.0)
            else:
                features[feat_name] = _DEFAULTS.get(feat_name, 0.0)
        return features

    def is_connected(self) -> bool:
        try:
            return self.r.ping()
        except redis.ConnectionError:
            return False