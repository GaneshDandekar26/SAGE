"""
SAGE ML Inference Service — 2-Stage Cascaded Pipeline

Stage 1: XGBoost Binary → Human vs Bot
Stage 2: Random Forest 3-Class → Flood vs Scraper vs Recon (bot only)

Endpoints:
    POST /predict           — Receive telemetry from Java Gateway, return verdict
    GET  /predict/{user_id} — Pull features from Redis, return verdict
    GET  /health            — Service and model health
    GET  /metrics           — Prometheus metrics
"""

from fastapi import FastAPI, HTTPException, Response
import joblib
import json
import os
import time
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
from assembler import FeatureAssembler, STAGE1_FEATURES, STAGE2_FEATURES, ALL_FEATURES
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel, Field


# ── Prometheus Metrics ───────────────────────────────────────────────
REQUESTS_TOTAL = Counter("sage_inference_requests_total", "Total prediction requests received")
THREATS_DETECTED_TOTAL = Counter("sage_inference_threats_detected_total", "Total requests classified as bots")
INFERENCE_LATENCY = Histogram("sage_inference_latency_seconds", "Time spent processing the ML prediction")
STAGE1_HUMAN_TOTAL = Counter("sage_stage1_human_total", "Requests classified as human by Stage 1")
STAGE1_BOT_TOTAL = Counter("sage_stage1_bot_total", "Requests classified as bot by Stage 1")

# ── Global Model State ──────────────────────────────────────────────
STAGE1_MODEL = None        # XGBoost: Human vs Bot
STAGE2_MODEL = None        # RandomForest: Flood vs Scraper vs Recon
STAGE2_ENCODER = None      # LabelEncoder for Stage 2 classes
STAGE1_THRESHOLD = 0.50    # Bot probability threshold (tuned for human safety)

assembler = FeatureAssembler(host="localhost", port=6379)

# Actions for graduated response
RESPONSE_ACTIONS = {
    "flood": "BAN",
    "scraper": "RATE_LIMIT",
    "recon": "CAPTCHA",
}


# ── Schemas ──────────────────────────────────────────────────────────

class GatewayTelemetry(BaseModel):
    """Full 22-feature payload from the Java Gateway."""
    session_id: str
    # Stage 1 — Behavioural features
    SAGE_InterArrival_CV: float = Field(default=1.0, description="CV of inter-request time gaps")
    SAGE_Timing_Entropy: float = Field(default=1.5, description="Shannon entropy of timing bins")
    SAGE_Pause_Ratio: float = Field(default=0.3, description="Fraction of gaps > 2s")
    SAGE_Burst_Score: float = Field(default=0.01, description="Burst clusters / session depth")
    SAGE_Backtrack_Ratio: float = Field(default=0.2, description="Revisits to seen endpoints")
    SAGE_Path_Entropy: float = Field(default=2.0, description="Shannon entropy of endpoint distribution")
    SAGE_Referral_Chain_Depth: float = Field(default=2.0, description="Mean hierarchical nav chain length")
    SAGE_Session_Depth: float = Field(default=1.0, description="Total requests in session")
    SAGE_Method_Diversity: float = Field(default=0.15, description="Unique HTTP methods / total")
    SAGE_Static_Asset_Ratio: float = Field(default=0.3, description="Static resource loads / total")
    SAGE_Error_Rate: float = Field(default=0.02, description="4xx+5xx responses / total")
    SAGE_Payload_Variance: float = Field(default=100.0, description="StdDev of request body sizes")
    # Stage 2 — Operational features
    SAGE_Request_Velocity: float = Field(default=1.0, description="Requests per minute")
    SAGE_Peak_Burst_RPS: float = Field(default=1.0, description="Max requests in any 1s window")
    SAGE_Velocity_Trend: float = Field(default=0.0, description="Slope of velocity over time")
    SAGE_Endpoint_Concentration: float = Field(default=0.3, description="Top-3 endpoints / total")
    SAGE_Cart_Ratio: float = Field(default=0.1, description="Cart+checkout / total")
    SAGE_Sequential_Traversal: float = Field(default=0.05, description="Consecutive ID traversal score")
    SAGE_Sensitive_Endpoint_Ratio: float = Field(default=0.01, description="Admin/config path fraction")
    SAGE_UA_Entropy: float = Field(default=0.0, description="User-Agent string entropy")
    SAGE_Header_Completeness: float = Field(default=0.9, description="Browser header presence score")
    SAGE_Response_Size_Variance: float = Field(default=200.0, description="StdDev of response sizes")


class InferenceResult(BaseModel):
    """Response returned to the Java Gateway."""
    session_id: str
    is_bot: bool
    bot_probability: float
    threat_class: str        # "Human", "Flood", "Scraper", "Recon"
    confidence: float
    action: str              # "ALLOW", "BAN", "RATE_LIMIT", "CAPTCHA", "MONITOR"
    processing_time_ms: float


# ── Lifespan & Model Loading ────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global STAGE1_MODEL, STAGE2_MODEL, STAGE2_ENCODER, STAGE1_THRESHOLD

    base_path = os.path.dirname(__file__)
    model_dir = os.path.abspath(os.path.join(base_path, "..", "models"))

    # Stage 1: Human vs Bot (XGBoost)
    stage1_path = os.path.join(model_dir, "human_vs_bot.pkl")
    threshold_path = os.path.join(model_dir, "human_vs_bot_threshold.json")

    if os.path.exists(stage1_path):
        STAGE1_MODEL = joblib.load(stage1_path)
        print(f"[+] Stage 1 model loaded: {stage1_path}")
    else:
        print(f"[!] WARNING: Stage 1 model not found at {stage1_path}")

    if os.path.exists(threshold_path):
        with open(threshold_path) as f:
            STAGE1_THRESHOLD = json.load(f).get("optimal_threshold", 0.50)
        print(f"[+] Stage 1 threshold: {STAGE1_THRESHOLD}")

    # Stage 2: Flood vs Scraper vs Recon (Random Forest)
    stage2_path = os.path.join(model_dir, "attack_classifier.pkl")
    encoder_path = os.path.join(model_dir, "attack_classifier_encoder.pkl")

    if os.path.exists(stage2_path):
        STAGE2_MODEL = joblib.load(stage2_path)
        print(f"[+] Stage 2 model loaded: {stage2_path}")
    else:
        print(f"[!] WARNING: Stage 2 model not found at {stage2_path}")

    if os.path.exists(encoder_path):
        STAGE2_ENCODER = joblib.load(encoder_path)
        print(f"[+] Stage 2 classes: {STAGE2_ENCODER.classes_.tolist()}")
    else:
        print(f"[!] WARNING: Stage 2 label encoder not found at {encoder_path}")

    ready = STAGE1_MODEL is not None and STAGE2_MODEL is not None
    print(f"\n[{'✓' if ready else '✗'}] SAGE 2-Stage Engine {'READY' if ready else 'DEGRADED'}")
    yield


app = FastAPI(
    lifespan=lifespan,
    title="SAGE ML Inference — 2-Stage Pipeline",
    description="Stage 1: Human vs Bot → Stage 2: Flood / Scraper / Recon",
)


# ── Core Inference Logic ─────────────────────────────────────────────

def run_two_stage_inference(feature_dict: dict) -> dict:
    """
    Execute the cascaded 2-stage pipeline.

    Returns:
        dict with is_bot, bot_probability, threat_class, confidence, action
    """
    # ──── STAGE 1: Human vs Bot ────
    X_stage1 = assembler.assemble_stage1(feature_dict)
    stage1_proba = STAGE1_MODEL.predict_proba(X_stage1)[0]
    bot_probability = float(stage1_proba[1])  # probability of class 1 (bot)

    if bot_probability < STAGE1_THRESHOLD:
        # Classified as HUMAN → allow traffic
        STAGE1_HUMAN_TOTAL.inc()
        return {
            "is_bot": False,
            "bot_probability": bot_probability,
            "threat_class": "Human",
            "confidence": float(1.0 - bot_probability),
            "action": "ALLOW",
        }

    # ──── STAGE 2: Bot Sub-Classification ────
    STAGE1_BOT_TOTAL.inc()
    THREATS_DETECTED_TOTAL.inc()

    X_stage2 = assembler.assemble_stage2(feature_dict)
    stage2_proba = STAGE2_MODEL.predict_proba(X_stage2)[0]
    predicted_idx = int(np.argmax(stage2_proba))
    stage2_confidence = float(stage2_proba[predicted_idx])

    threat_class = str(STAGE2_ENCODER.inverse_transform([predicted_idx])[0])

    # Determine graduated response action
    if bot_probability >= 0.80:
        action = RESPONSE_ACTIONS.get(threat_class, "BAN")
    elif bot_probability >= 0.60:
        action = "MONITOR"
    else:
        action = "ALLOW"

    return {
        "is_bot": True,
        "bot_probability": bot_probability,
        "threat_class": threat_class.capitalize(),
        "confidence": stage2_confidence,
        "action": action,
    }


# ── Endpoints ────────────────────────────────────────────────────────

@app.post("/predict", response_model=InferenceResult)
def predict_anomaly(data: GatewayTelemetry):
    """
    Receives real-time telemetry from the Java Gateway.
    Runs the 2-stage pipeline and returns a verdict.
    """
    if STAGE1_MODEL is None or STAGE2_MODEL is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    start_time = time.perf_counter()
    REQUESTS_TOTAL.inc()

    try:
        # Convert Pydantic model to feature dict
        feature_dict = assembler.assemble_from_payload(data.model_dump())

        result = run_two_stage_inference(feature_dict)

        processing_ms = round((time.perf_counter() - start_time) * 1000, 3)
        INFERENCE_LATENCY.observe(time.perf_counter() - start_time)

        return InferenceResult(
            session_id=data.session_id,
            is_bot=result["is_bot"],
            bot_probability=round(result["bot_probability"], 4),
            threat_class=result["threat_class"],
            confidence=round(result["confidence"], 4),
            action=result["action"],
            processing_time_ms=processing_ms,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predict/{user_id}")
async def predict_bot(user_id: str):
    """
    Pull features from Redis for a user and run the 2-stage pipeline.
    """
    if STAGE1_MODEL is None or STAGE2_MODEL is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    start_time = time.perf_counter()
    REQUESTS_TOTAL.inc()

    try:
        feature_dict = assembler.assemble_full(user_id)
        result = run_two_stage_inference(feature_dict)

        processing_ms = round((time.perf_counter() - start_time) * 1000, 3)
        INFERENCE_LATENCY.observe(time.perf_counter() - start_time)

        return {
            "user_id": user_id,
            "is_bot": result["is_bot"],
            "bot_probability": round(result["bot_probability"], 4),
            "threat_class": result["threat_class"],
            "confidence": round(result["confidence"], 4),
            "action": result["action"],
            "processing_time_ms": processing_ms,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
def health_check():
    stage1_ok = STAGE1_MODEL is not None
    stage2_ok = STAGE2_MODEL is not None and STAGE2_ENCODER is not None

    return {
        "status": "operational" if (stage1_ok and stage2_ok) else "degraded",
        "stage1_loaded": stage1_ok,
        "stage2_loaded": stage2_ok,
        "stage1_threshold": STAGE1_THRESHOLD,
        "stage2_classes": STAGE2_ENCODER.classes_.tolist() if STAGE2_ENCODER else [],
        "stage1_features": STAGE1_FEATURES,
        "stage2_features": STAGE2_FEATURES,
        "pipeline": "2-stage cascaded (Human→Bot→Flood/Scraper/Recon)",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)