# ml_pipeline/features/timing_features.py
"""
Stage 1 – Temporal Micro-Pattern Features (Human vs Bot)

These four features capture timing irregularity that separates real humans
(who pause to read, think, and click at varied speeds) from bots (who
operate at fixed or near-fixed intervals).

All calculations are performed inside Redis Lua scripts so the entire
read-modify-write cycle is atomic and takes < 1 ms at p99.
"""

import math
import redis
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Bin edges for Timing Entropy (milliseconds)
_ENTROPY_BIN_EDGES = [0, 100, 500, 2000, 5000, float("inf")]
_NUM_BINS = len(_ENTROPY_BIN_EDGES) - 1  # 5 bins

# Thresholds
_PAUSE_THRESHOLD_MS = 2000   # > 2 s counts as a "reading pause"
_BURST_WINDOW_MS = 500       # 3+ requests within 500 ms = 1 burst cluster
_SLIDING_WINDOW = 20         # last N inter-arrival gaps to consider
_TTL_SECONDS = 86400         # 24 h session expiry


class TimingFeaturesCalculator:
    """
    Computes four timing features from the stream of request timestamps:
        1. SAGE_InterArrival_CV   – Coefficient of Variation of gaps
        2. SAGE_Timing_Entropy    – Shannon entropy across time bins
        3. SAGE_Pause_Ratio       – Fraction of gaps > 2 s
        4. SAGE_Burst_Score       – Burst-cluster-count / session-depth
    """

    def __init__(self, redis_host="localhost", redis_port=6379):
        self.redis_client = redis.Redis(
            host=redis_host, port=redis_port, decode_responses=True
        )

        # ------------------------------------------------------------------
        # Lua script: update interval list and return raw data for Python
        # to compute the four features.  Keeping heavy math (entropy, CV)
        # in Python keeps the Lua script simple and debuggable while the
        # Redis round-trip is still a single atomic call.
        # ------------------------------------------------------------------
        self.lua_script = self.redis_client.register_script("""
            local key_prefix  = KEYS[1]
            local current_ts  = tonumber(ARGV[1])
            local window_size = tonumber(ARGV[2])
            local ttl         = tonumber(ARGV[3])

            local last_ts_key   = key_prefix .. ':last_ts'
            local intervals_key = key_prefix .. ':intervals'
            local depth_key     = key_prefix .. ':depth'

            -- Increment session depth
            local depth = redis.call('INCR', depth_key)
            redis.call('EXPIRE', depth_key, ttl)

            -- Get previous timestamp
            local last_ts = redis.call('GET', last_ts_key)
            redis.call('SET', last_ts_key, current_ts, 'EX', ttl)

            if not last_ts then
                return { tostring(-1), tostring(depth) }
            end

            -- Calculate delta (ms)
            local delta = math.abs(current_ts - tonumber(last_ts))

            -- Push to sliding window
            redis.call('LPUSH', intervals_key, delta)
            redis.call('LTRIM', intervals_key, 0, window_size - 1)
            redis.call('EXPIRE', intervals_key, ttl)

            -- Return all intervals + depth so Python can compute features
            local intervals = redis.call('LRANGE', intervals_key, 0, -1)
            table.insert(intervals, 1, tostring(depth))  -- prepend depth
            return intervals
        """)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate(self, user_id: str, timestamp_val) -> dict | None:
        """
        Call on every incoming request.  Returns a dict with the four
        timing features, or None if not enough data yet (< 3 intervals).
        """
        current_ts_ms = self._extract_ms(timestamp_val)
        if current_ts_ms is None:
            return None

        redis_key = f"features:timing:{user_id}"

        try:
            raw = self.lua_script(
                keys=[redis_key],
                args=[current_ts_ms, _SLIDING_WINDOW, _TTL_SECONDS],
            )
        except Exception as e:
            logger.error(f"Redis error for timing features ({user_id}): {e}")
            return None

        # First call – no interval yet
        if len(raw) == 2 and raw[0] == "-1":
            return None

        # Parse: first element is depth, rest are interval strings
        depth = int(raw[0])
        intervals = [float(v) for v in raw[1:]]

        if len(intervals) < 3:
            return None  # Need ≥ 3 gaps for meaningful stats

        features = {
            "SAGE_InterArrival_CV": self._cv(intervals),
            "SAGE_Timing_Entropy": self._timing_entropy(intervals),
            "SAGE_Pause_Ratio": self._pause_ratio(intervals),
            "SAGE_Burst_Score": self._burst_score(intervals, depth),
        }

        logger.debug(
            f"[TIMING] User {user_id} | CV={features['SAGE_InterArrival_CV']:.4f} "
            f"Entropy={features['SAGE_Timing_Entropy']:.4f} "
            f"PauseR={features['SAGE_Pause_Ratio']:.4f} "
            f"BurstS={features['SAGE_Burst_Score']:.4f}"
        )
        return features

    # ------------------------------------------------------------------
    # Feature math (pure Python – no Redis calls)
    # ------------------------------------------------------------------

    @staticmethod
    def _cv(intervals: list[float]) -> float:
        """Coefficient of Variation = std / mean.  High = human, Low = bot."""
        n = len(intervals)
        mean = sum(intervals) / n
        if mean == 0:
            return 0.0
        variance = sum((x - mean) ** 2 for x in intervals) / n
        return math.sqrt(variance) / mean

    @staticmethod
    def _timing_entropy(intervals: list[float]) -> float:
        """Shannon entropy over 5 time-gap bins.  High entropy = human."""
        bins = [0] * _NUM_BINS
        for gap in intervals:
            for i in range(_NUM_BINS):
                if gap < _ENTROPY_BIN_EDGES[i + 1]:
                    bins[i] += 1
                    break

        total = len(intervals)
        entropy = 0.0
        for count in bins:
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        return entropy

    @staticmethod
    def _pause_ratio(intervals: list[float]) -> float:
        """Fraction of gaps > 2 seconds.  Higher = more human-like."""
        pauses = sum(1 for gap in intervals if gap > _PAUSE_THRESHOLD_MS)
        return pauses / len(intervals)

    @staticmethod
    def _burst_score(intervals: list[float], session_depth: int) -> float:
        """
        Count burst clusters (3+ requests within 500 ms window)
        normalized by session depth.  High ratio = bot.
        """
        if session_depth <= 0:
            return 0.0

        burst_count = 0
        consecutive_fast = 0
        for gap in intervals:
            if gap <= _BURST_WINDOW_MS:
                consecutive_fast += 1
                if consecutive_fast >= 2:  # 3 requests = 2 consecutive fast gaps
                    burst_count += 1
                    consecutive_fast = 0  # reset to count next cluster
            else:
                consecutive_fast = 0

        return burst_count / session_depth

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_ms(timestamp_val) -> int | None:
        try:
            if isinstance(timestamp_val, (int, float)):
                return int(timestamp_val)
            if isinstance(timestamp_val, str):
                clean = timestamp_val.replace("Z", "+00:00")
                dt = datetime.fromisoformat(clean)
                return int(dt.timestamp() * 1000)
        except Exception as e:
            logger.error(f"Error parsing timestamp {timestamp_val}: {e}")
        return None
