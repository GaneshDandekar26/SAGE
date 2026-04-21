"""
SAGE ML Inference Service — 2-Stage Cascaded Pipeline
=====================================================

Flow:
    Request → Stage 1 (Human vs Bot)
              ├─ Human  → ALLOW  (return immediately, skip Stage 2)
              └─ Bot    → Stage 2 (Flood / Scraper / Recon)
                          ├─ Flood   → BAN
                          ├─ Scraper → RATE_LIMIT
                          └─ Recon   → CAPTCHA

Threshold:
    Configurable via:
      1. models/human_vs_bot_threshold.json  (persisted, loaded on startup)
      2. PUT /config/threshold               (runtime hot-reload, no restart)
      3. Environment variable SAGE_BOT_THRESHOLD (overrides file on startup)

    Higher threshold → more conservative (fewer humans blocked, more bots leak)
    Lower  threshold → more aggressive  (catches more bots, risks blocking humans)

    Recommended range: 0.30 – 0.60
    Default: 0.50

Endpoints:
    POST /predict             — full telemetry from Java Gateway
    GET  /predict/{user_id}   — pull features from Redis
    GET  /health              — service + model health
    GET  /metrics             — Prometheus metrics
    GET  /config/threshold    — read current threshold
    PUT  /config/threshold    — update threshold at runtime
"""

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from prometheus_client import (
    Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST,
)

import joblib
import json
import logging
import numpy as np
import os
import pandas as pd
import time

from assembler import (
    FeatureAssembler,
    STAGE1_FEATURES,
    STAGE2_FEATURES,
    ALL_FEATURES,
)

# ── Logging ──────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("sage.inference")

# ── Prometheus Metrics ───────────────────────────────────────────────

REQ_TOTAL       = Counter("sage_requests_total", "Total prediction requests")
BOT_TOTAL       = Counter("sage_bots_detected_total", "Total bots detected")
HUMAN_TOTAL     = Counter("sage_humans_allowed_total", "Total humans allowed")
LATENCY         = Histogram("sage_latency_seconds", "End-to-end prediction latency")
THRESHOLD_GAUGE = Gauge("sage_threshold", "Current bot-detection threshold")

STAGE2_CLASS_COUNTER = Counter(
    "sage_stage2_class_total",
    "Stage 2 class predictions",
    ["threat_class"],
)

# ── Global State ─────────────────────────────────────────────────────

STAGE1_MODEL   = None
STAGE2_MODEL   = None
STAGE2_ENCODER = None
BOT_THRESHOLD  = 0.50

assembler = FeatureAssembler(host="localhost", port=6379)

# Graduated response mapping
RESPONSE_MAP = {
    "flood":   "BAN",
    "scraper": "RATE_LIMIT",
    "recon":   "CAPTCHA",
}


# ── Schemas ──────────────────────────────────────────────────────────

class GatewayTelemetry(BaseModel):
    """22-feature payload from the Java Gateway."""
    session_id: str
    # — Stage 1: Behavioural features (12) —
    SAGE_InterArrival_CV:       float = Field(default=1.0)
    SAGE_Timing_Entropy:        float = Field(default=1.5)
    SAGE_Pause_Ratio:           float = Field(default=0.3)
    SAGE_Burst_Score:           float = Field(default=0.01)
    SAGE_Backtrack_Ratio:       float = Field(default=0.2)
    SAGE_Path_Entropy:          float = Field(default=2.0)
    SAGE_Referral_Chain_Depth:  float = Field(default=2.0)
    SAGE_Session_Depth:         float = Field(default=1.0)
    SAGE_Method_Diversity:      float = Field(default=0.15)
    SAGE_Static_Asset_Ratio:    float = Field(default=0.3)
    SAGE_Error_Rate:            float = Field(default=0.02)
    SAGE_Payload_Variance:      float = Field(default=100.0)
    # — Stage 2: Operational features (10) —
    SAGE_Request_Velocity:       float = Field(default=1.0)
    SAGE_Peak_Burst_RPS:         float = Field(default=1.0)
    SAGE_Velocity_Trend:         float = Field(default=0.0)
    SAGE_Endpoint_Concentration: float = Field(default=0.3)
    SAGE_Cart_Ratio:             float = Field(default=0.1)
    SAGE_Sequential_Traversal:   float = Field(default=0.05)
    SAGE_Sensitive_Endpoint_Ratio: float = Field(default=0.01)
    SAGE_UA_Entropy:             float = Field(default=0.0)
    SAGE_Header_Completeness:    float = Field(default=0.9)
    SAGE_Response_Size_Variance: float = Field(default=200.0)


class InferenceResult(BaseModel):
    """Response returned to the Java Gateway."""
    session_id:        str
    is_bot:            bool
    bot_probability:   float
    threat_class:      str       # "Human" | "Flood" | "Scraper" | "Recon"
    confidence:        float
    action:            str       # "ALLOW" | "BAN" | "RATE_LIMIT" | "CAPTCHA" | "MONITOR"
    processing_time_ms: float
    stage1_threshold:  float     # current threshold (for observability)


class ThresholdUpdate(BaseModel):
    """Body for PUT /config/threshold."""
    threshold: float = Field(ge=0.01, le=0.99, description="Bot probability threshold")


# ── Model Loading ────────────────────────────────────────────────────

def _load_models(model_dir: str) -> None:
    """Load both models and threshold from disk."""
    global STAGE1_MODEL, STAGE2_MODEL, STAGE2_ENCODER, BOT_THRESHOLD

    # Stage 1
    s1_path = os.path.join(model_dir, "human_vs_bot.pkl")
    if os.path.exists(s1_path):
        STAGE1_MODEL = joblib.load(s1_path)
        logger.info(f"Stage 1 loaded: {s1_path}")
    else:
        logger.warning(f"Stage 1 model NOT FOUND: {s1_path}")

    # Stage 2
    s2_path = os.path.join(model_dir, "attack_classifier.pkl")
    enc_path = os.path.join(model_dir, "attack_classifier_encoder.pkl")
    if os.path.exists(s2_path):
        STAGE2_MODEL = joblib.load(s2_path)
        logger.info(f"Stage 2 loaded: {s2_path}")
    else:
        logger.warning(f"Stage 2 model NOT FOUND: {s2_path}")

    if os.path.exists(enc_path):
        STAGE2_ENCODER = joblib.load(enc_path)
        logger.info(f"Stage 2 classes: {STAGE2_ENCODER.classes_.tolist()}")

    # Threshold: env var → file → default
    env_threshold = os.environ.get("SAGE_BOT_THRESHOLD")
    if env_threshold:
        BOT_THRESHOLD = float(env_threshold)
        logger.info(f"Threshold from env: {BOT_THRESHOLD}")
    else:
        thr_path = os.path.join(model_dir, "human_vs_bot_threshold.json")
        if os.path.exists(thr_path):
            with open(thr_path) as f:
                BOT_THRESHOLD = json.load(f).get("optimal_threshold", 0.50)
            logger.info(f"Threshold from file: {BOT_THRESHOLD}")
        else:
            logger.info(f"Threshold default: {BOT_THRESHOLD}")

    THRESHOLD_GAUGE.set(BOT_THRESHOLD)


@asynccontextmanager
async def lifespan(app: FastAPI):
    base = os.path.dirname(__file__)
    model_dir = os.path.abspath(os.path.join(base, "..", "models"))
    _load_models(model_dir)

    ready = STAGE1_MODEL is not None and STAGE2_MODEL is not None
    logger.info(f"SAGE 2-Stage Engine: {'READY' if ready else 'DEGRADED'}")
    yield


app = FastAPI(
    lifespan=lifespan,
    title="SAGE ML Inference — 2-Stage Pipeline",
    version="2.0.0",
)


# ── Core Inference ───────────────────────────────────────────────────

def _predict(feature_dict: dict) -> dict:
    """
    Execute the 2-stage cascade.

    STAGE 1  →  bot_probability < threshold?
                    YES → return Human (ALLOW)
                    NO  → continue to Stage 2

    STAGE 2  →  classify bot type → graduated response
    """
    # ── Stage 1: Human vs Bot ────────────────────────────────────────
    X1 = assembler.assemble_stage1(feature_dict)
    proba = STAGE1_MODEL.predict_proba(X1)[0]
    bot_prob = float(proba[1])

    if bot_prob < BOT_THRESHOLD:
        HUMAN_TOTAL.inc()
        return {
            "is_bot": False,
            "bot_probability": bot_prob,
            "threat_class": "Human",
            "confidence": round(1.0 - bot_prob, 4),
            "action": "ALLOW",
        }

    # ── Stage 2: Attack Classification ───────────────────────────────
    BOT_TOTAL.inc()
    X2 = assembler.assemble_stage2(feature_dict)
    s2_proba = STAGE2_MODEL.predict_proba(X2)[0]
    pred_idx = int(np.argmax(s2_proba))
    s2_conf = float(s2_proba[pred_idx])
    threat = str(STAGE2_ENCODER.inverse_transform([pred_idx])[0])

    STAGE2_CLASS_COUNTER.labels(threat_class=threat).inc()

    # Graduated response
    if bot_prob >= 0.80:
        action = RESPONSE_MAP.get(threat, "BAN")
    elif bot_prob >= 0.60:
        action = "MONITOR"
    else:
        action = "ALLOW"

    return {
        "is_bot": True,
        "bot_probability": bot_prob,
        "threat_class": threat.capitalize(),
        "confidence": round(s2_conf, 4),
        "action": action,
    }


# ── Endpoints ────────────────────────────────────────────────────────

@app.post("/predict", response_model=InferenceResult)
def predict_from_payload(data: GatewayTelemetry):
    """Receive full telemetry from the Java Gateway and return a verdict."""
    if STAGE1_MODEL is None or STAGE2_MODEL is None:
        raise HTTPException(503, "Models not loaded")

    t0 = time.perf_counter()
    REQ_TOTAL.inc()

    try:
        features = assembler.assemble_from_payload(data.model_dump())
        result = _predict(features)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)
        LATENCY.observe(time.perf_counter() - t0)

        return InferenceResult(
            session_id=data.session_id,
            processing_time_ms=elapsed_ms,
            stage1_threshold=BOT_THRESHOLD,
            **result,
        )
    except Exception as e:
        logger.exception("Prediction error")
        raise HTTPException(500, str(e))


@app.get("/predict/{user_id}")
async def predict_from_redis(user_id: str):
    """Pull features from Redis and return a verdict."""
    if STAGE1_MODEL is None or STAGE2_MODEL is None:
        raise HTTPException(503, "Models not loaded")

    t0 = time.perf_counter()
    REQ_TOTAL.inc()

    try:
        features = assembler.assemble_full(user_id)
        result = _predict(features)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)
        LATENCY.observe(time.perf_counter() - t0)

        return {
            "user_id": user_id,
            "processing_time_ms": elapsed_ms,
            "stage1_threshold": BOT_THRESHOLD,
            **result,
        }
    except Exception as e:
        logger.exception("Prediction error")
        raise HTTPException(500, str(e))


# ── Threshold Configuration ──────────────────────────────────────────

@app.get("/config/threshold")
def get_threshold():
    """
    Read the current bot-detection threshold.

    Adjusting the threshold:
      - INCREASE (e.g. 0.70) → more lenient, fewer humans blocked,
        but some bots may slip through.
      - DECREASE (e.g. 0.30) → more aggressive, catches more bots,
        but risks blocking borderline humans.

    Recommendation: start at 0.50 and lower **only** if human recall
    is already ≥ 95% on your evaluation set.
    """
    return {
        "threshold": BOT_THRESHOLD,
        "description": "bot_probability >= threshold → classified as bot",
        "how_to_adjust": {
            "runtime": "PUT /config/threshold  {\"threshold\": 0.40}",
            "startup_env": "export SAGE_BOT_THRESHOLD=0.40",
            "startup_file": "models/human_vs_bot_threshold.json",
        },
    }


@app.put("/config/threshold")
def set_threshold(body: ThresholdUpdate):
    """Update the bot-detection threshold without restarting the service."""
    global BOT_THRESHOLD
    old = BOT_THRESHOLD
    BOT_THRESHOLD = body.threshold
    THRESHOLD_GAUGE.set(BOT_THRESHOLD)

    # Persist to disk so it survives restarts
    base = os.path.dirname(__file__)
    thr_path = os.path.abspath(
        os.path.join(base, "..", "models", "human_vs_bot_threshold.json")
    )
    try:
        with open(thr_path, "w") as f:
            json.dump({"optimal_threshold": BOT_THRESHOLD}, f, indent=2)
    except OSError:
        logger.warning("Could not persist threshold to disk")

    logger.info(f"Threshold changed: {old} → {BOT_THRESHOLD}")
    return {
        "previous": old,
        "current": BOT_THRESHOLD,
        "persisted": True,
    }


# ── Observability ────────────────────────────────────────────────────

@app.get("/metrics")
async def prometheus_metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
def health():
    s1 = STAGE1_MODEL is not None
    s2 = STAGE2_MODEL is not None and STAGE2_ENCODER is not None
    return {
        "status": "operational" if (s1 and s2) else "degraded",
        "stage1_loaded": s1,
        "stage2_loaded": s2,
        "threshold": BOT_THRESHOLD,
        "stage2_classes": STAGE2_ENCODER.classes_.tolist() if STAGE2_ENCODER else [],
        "feature_counts": {
            "stage1": len(STAGE1_FEATURES),
            "stage2": len(STAGE2_FEATURES),
            "total": len(ALL_FEATURES),
        },
    }


# ── Entrypoint ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)