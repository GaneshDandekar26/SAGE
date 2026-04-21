"""
SAGE ML Feature Engineering Pipeline — Kafka Consumer

Consumes gateway telemetry events from Kafka and computes all 22 features
in real-time using the Redis-backed feature calculators.

Feature calculators:
  - TimingFeaturesCalculator     → InterArrival_CV, Timing_Entropy, Pause_Ratio, Burst_Score
  - NavigationFeaturesCalculator → Backtrack_Ratio, Path_Entropy, Referral_Chain_Depth
  - ContentFeaturesCalculator    → Method_Diversity, Static_Asset_Ratio, Error_Rate, Payload_Variance
  - AttackFeaturesCalculator     → Peak_Burst_RPS, Velocity_Trend, Sensitive_Endpoint_Ratio,
                                   UA_Entropy, Header_Completeness, Response_Size_Variance
  - SessionDepthCalculator       → Session_Depth (existing)
  - RequestVelocityCalculator    → Request_Velocity (existing)
  - EndpointDiversityCalculator  → Endpoint_Concentration (existing)
"""

import json
import logging
import signal
import sys
import os
import uuid

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.session_depth import SessionDepthCalculator
from features.request_velocity import RequestVelocityCalculator
from features.endpoint_diversity import EndpointDiversityCalculator
from features.temporal_variance import TemporalVarianceCalculator
from features.timing_features import TimingFeaturesCalculator
from features.navigation_features import NavigationFeaturesCalculator
from features.content_features import ContentFeaturesCalculator
from features.attack_features import AttackFeaturesCalculator
from confluent_kafka import Consumer, KafkaError, KafkaException

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class GatewayEventConsumer:
    def __init__(self, bootstrap_servers="localhost:9092", topic="gateway-telemetry", group_id="feature-engineering"):
        conf = {
            "bootstrap.servers": bootstrap_servers,
            "group.id": group_id,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": True,
        }

        self.consumer = Consumer(conf)
        self.topic = topic
        self.running = True

        self.consumer.subscribe([self.topic])
        logger.info(f"Initialized Kafka consumer for topic: {self.topic}")

    def consume_events(self, process_callback):
        """Main polling loop."""
        try:
            while self.running:
                msg = self.consumer.poll(timeout=1.0)

                if msg is None:
                    continue

                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        raise KafkaException(msg.error())

                try:
                    raw_value = msg.value().decode("utf-8")
                    event_data = json.loads(raw_value)
                    process_callback(event_data)
                except json.JSONDecodeError:
                    logger.error("Failed to parse JSON. Skipping message.")
                except Exception as e:
                    logger.error(f"Error processing event payload: {e}")

        except KeyboardInterrupt:
            logger.info("Manual interrupt detected (Ctrl+C). Shutting down...")
        finally:
            self.close()

    def stop(self, signum=None, frame=None):
        """Signal handler for graceful shutdown."""
        self.running = False

    def close(self):
        """Graceful shutdown."""
        logger.info("Closing Kafka consumer connection...")
        self.consumer.close()


if __name__ == "__main__":
    consumer = GatewayEventConsumer()

    # ── Initialize ALL Feature Calculators ──
    # Existing (kept)
    depth_calculator = SessionDepthCalculator()
    velocity_calculator = RequestVelocityCalculator()
    diversity_calculator = EndpointDiversityCalculator()
    variance_calculator = TemporalVarianceCalculator()

    # New Stage 1 calculators
    timing_calculator = TimingFeaturesCalculator()
    navigation_calculator = NavigationFeaturesCalculator()
    content_calculator = ContentFeaturesCalculator()

    # New Stage 2 calculator
    attack_calculator = AttackFeaturesCalculator()

    signal.signal(signal.SIGINT, consumer.stop)
    signal.signal(signal.SIGTERM, consumer.stop)

    def process_event(event):
        """
        Process a single gateway telemetry event and compute all features.

        Expected event JSON structure from the Java Gateway:
        {
            "userId": "192.168.1.100",
            "sessionId": "abc123",
            "requestId": "uuid",
            "timestamp": 1712345678000,
            "request": {
                "method": "GET",
                "path": "/api/products/123",
                "headers": {"Accept": "...", "Accept-Language": "..."},
                "userAgent": "Mozilla/5.0 ...",
                "contentLength": 0
            },
            "response": {
                "status": 200,
                "contentLength": 4567
            }
        }
        """
        user = event.get("userId", "unknown_user")
        session = event.get("sessionId", user)
        req_id = event.get("requestId") or str(uuid.uuid4())
        timestamp = event.get("timestamp", "")

        request_data = event.get("request", {})
        response_data = event.get("response", {})

        path = request_data.get("path", "")
        method = request_data.get("method", "GET")
        user_agent = request_data.get("userAgent", "")
        headers = request_data.get("headers", {})
        request_size = float(request_data.get("contentLength", 0))
        response_status = int(response_data.get("status", 200))
        response_size = float(response_data.get("contentLength", 0))

        if not path or not timestamp:
            return

        print("-" * 50)
        logger.info(f"Processing: user={user} path={path} method={method}")

        # ── EXISTING calculators (kept for backwards compatibility) ──
        diversity_calculator.calculate(user, path)
        variance_calculator.calculate(user, timestamp)
        depth_calculator.calculate(session)
        velocity_calculator.calculate(user, timestamp, req_id)

        # ── NEW Stage 1: Timing Features ──
        timing_result = timing_calculator.calculate(user, timestamp)
        if timing_result:
            logger.info(f"  Timing: CV={timing_result['SAGE_InterArrival_CV']:.4f} "
                        f"Entropy={timing_result['SAGE_Timing_Entropy']:.4f} "
                        f"PauseR={timing_result['SAGE_Pause_Ratio']:.4f} "
                        f"BurstS={timing_result['SAGE_Burst_Score']:.4f}")

        # ── NEW Stage 1: Navigation Features ──
        nav_result = navigation_calculator.calculate(user, path)
        if nav_result:
            logger.info(f"  Nav: Backtrack={nav_result['SAGE_Backtrack_Ratio']:.4f} "
                        f"PathEnt={nav_result['SAGE_Path_Entropy']:.4f} "
                        f"ChainD={nav_result['SAGE_Referral_Chain_Depth']:.4f}")

        # ── NEW Stage 1: Content Features ──
        content_result = content_calculator.calculate(
            user, method, path, response_status, request_size
        )
        if content_result:
            logger.info(f"  Content: MethodDiv={content_result['SAGE_Method_Diversity']:.4f} "
                        f"StaticR={content_result['SAGE_Static_Asset_Ratio']:.4f} "
                        f"ErrorR={content_result['SAGE_Error_Rate']:.4f} "
                        f"PayloadVar={content_result['SAGE_Payload_Variance']:.2f}")

        # ── NEW Stage 2: Attack Features ──
        ts_ms = int(timestamp) if isinstance(timestamp, (int, float)) else 0
        attack_result = attack_calculator.calculate(
            user, ts_ms, path, user_agent, headers, response_size
        )
        if attack_result:
            logger.info(f"  Attack: PeakRPS={attack_result['SAGE_Peak_Burst_RPS']:.0f} "
                        f"VelTrend={attack_result['SAGE_Velocity_Trend']:.4f} "
                        f"SensitiveR={attack_result['SAGE_Sensitive_Endpoint_Ratio']:.4f}")

    logger.info("=" * 50)
    logger.info("SAGE 2-Stage ML Feature Pipeline — All engines online")
    logger.info(f" Timing:     InterArrival_CV, Timing_Entropy, Pause_Ratio, Burst_Score")
    logger.info(f" Navigation: Backtrack_Ratio, Path_Entropy, Referral_Chain_Depth")
    logger.info(f" Content:    Method_Diversity, Static_Asset_Ratio, Error_Rate, Payload_Variance")
    logger.info(f" Attack:     Peak_Burst_RPS, Velocity_Trend, Sensitive_Endpoint_Ratio,")
    logger.info(f"             UA_Entropy, Header_Completeness, Response_Size_Variance")
    logger.info(f" Legacy:     Session_Depth, Request_Velocity, Endpoint_Diversity, Temporal_Variance")
    logger.info("=" * 50)
    consumer.consume_events(process_event)