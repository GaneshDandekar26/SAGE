# ml_pipeline/features/attack_features.py
"""
Stage 2 – Attack Classification Features (Flood vs Scraper vs Recon)

These features are only computed for traffic already classified as "Bot"
by Stage 1.  They identify the *operational fingerprint* of the attack
to enable graduated response (BAN / RATE_LIMIT / CAPTCHA).

Features:
    1. SAGE_Peak_Burst_RPS           – Max requests/second in any 1s window
    2. SAGE_Velocity_Trend           – Slope of req/min over time
    3. SAGE_Sensitive_Endpoint_Ratio – Fraction targeting admin/config paths
    4. SAGE_UA_Entropy               – Shannon entropy of User-Agent strings
    5. SAGE_Header_Completeness      – Score from standard browser headers
    6. SAGE_Response_Size_Variance   – StdDev of response body sizes
"""

import math
import redis
import logging

logger = logging.getLogger(__name__)

_TTL_SECONDS = 86400
_SLIDING_WINDOW = 50

# Endpoints that indicate reconnaissance / probing behaviour
_SENSITIVE_PATTERNS = [
    "/admin", "/login", "/wp-admin", "/wp-login",
    "/.env", "/config", "/debug", "/swagger",
    "/graphql", "/api-docs", "/phpinfo", "/server-status",
    "/actuator", "/.git", "/backup", "/console",
]

# Standard headers that a real browser always sends
_BROWSER_HEADERS = [
    "Accept",
    "Accept-Language",
    "Accept-Encoding",
    "Connection",
]


class AttackFeaturesCalculator:
    """
    Computes 6 operational features that discriminate between
    Flood, Scraper, and Recon attack patterns.
    """

    def __init__(self, redis_host="localhost", redis_port=6379):
        self.redis_client = redis.Redis(
            host=redis_host, port=redis_port, decode_responses=True
        )

        # Lua: track per-second request counts, velocity trend buckets,
        # sensitive endpoint hits, UA frequency, and response sizes
        self.lua_script = self.redis_client.register_script("""
            local key_prefix    = KEYS[1]
            local current_sec   = ARGV[1]           -- epoch second
            local is_sensitive  = tonumber(ARGV[2])  -- 1 or 0
            local user_agent    = ARGV[3]
            local header_score  = tonumber(ARGV[4])  -- 0-4
            local response_size = tonumber(ARGV[5])
            local window_size   = tonumber(ARGV[6])
            local ttl           = tonumber(ARGV[7])

            local rps_key       = key_prefix .. ':rps'
            local peak_rps_key  = key_prefix .. ':peak_rps'
            local total_key     = key_prefix .. ':total'
            local sensitive_key = key_prefix .. ':sensitive'
            local ua_freq_key   = key_prefix .. ':ua_freq'
            local resp_sizes_key = key_prefix .. ':resp_sizes'
            local vel_buckets_key = key_prefix .. ':vel_buckets'

            -- 1. RPS tracking: increment counter for current second
            local current_rps = redis.call('HINCRBY', rps_key, current_sec, 1)
            redis.call('EXPIRE', rps_key, 120)  -- keep 2 min of second-buckets

            -- Update peak RPS if this second is the highest
            local peak = tonumber(redis.call('GET', peak_rps_key) or '0')
            if current_rps > peak then
                redis.call('SET', peak_rps_key, current_rps, 'EX', ttl)
            end

            -- 2. Total requests
            local total = redis.call('INCR', total_key)
            redis.call('EXPIRE', total_key, ttl)

            -- 3. Velocity trend: per-minute bucket for slope calculation
            local minute_bucket = math.floor(tonumber(current_sec) / 60)
            redis.call('HINCRBY', vel_buckets_key, minute_bucket, 1)
            redis.call('EXPIRE', vel_buckets_key, ttl)

            -- 4. Sensitive endpoint counter
            if is_sensitive == 1 then
                redis.call('INCR', sensitive_key)
            else
                if redis.call('EXISTS', sensitive_key) == 0 then
                    redis.call('SET', sensitive_key, '0', 'EX', ttl)
                end
            end
            redis.call('EXPIRE', sensitive_key, ttl)
            local sensitive_count = tonumber(redis.call('GET', sensitive_key) or '0')

            -- 5. User-Agent frequency tracking
            redis.call('HINCRBY', ua_freq_key, user_agent, 1)
            redis.call('EXPIRE', ua_freq_key, ttl)

            -- 6. Response size sliding window
            redis.call('LPUSH', resp_sizes_key, response_size)
            redis.call('LTRIM', resp_sizes_key, 0, window_size - 1)
            redis.call('EXPIRE', resp_sizes_key, ttl)

            -- Gather return data
            local peak_rps_val = tonumber(redis.call('GET', peak_rps_key) or '0')
            local ua_pairs = redis.call('HGETALL', ua_freq_key)
            local resp_sizes = redis.call('LRANGE', resp_sizes_key, 0, -1)
            local vel_pairs = redis.call('HGETALL', vel_buckets_key)

            -- Encode: total, peak_rps, sensitive_count, header_score, then markers
            local result = {
                tostring(total),
                tostring(peak_rps_val),
                tostring(sensitive_count),
                tostring(header_score)
            }
            -- UA freq pairs
            table.insert(result, '|||UA|||')
            for _, v in ipairs(ua_pairs) do table.insert(result, v) end
            -- Response sizes
            table.insert(result, '|||RESP|||')
            for _, v in ipairs(resp_sizes) do table.insert(result, v) end
            -- Velocity buckets
            table.insert(result, '|||VEL|||')
            for _, v in ipairs(vel_pairs) do table.insert(result, v) end

            return result
        """)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate(
        self,
        user_id: str,
        timestamp_ms: int,
        path: str,
        user_agent: str,
        headers: dict[str, str],
        response_size: float,
    ) -> dict | None:
        """
        Call for each request from a confirmed bot (Stage 2 only).

        Returns dict with 6 attack-classification features.
        """
        current_sec = str(timestamp_ms // 1000)
        is_sensitive = 1 if self._is_sensitive(path) else 0
        header_score = self._score_headers(headers)
        ua = user_agent or "unknown"
        redis_key = f"features:attack:{user_id}"

        try:
            raw = self.lua_script(
                keys=[redis_key],
                args=[
                    current_sec,
                    is_sensitive,
                    ua,
                    header_score,
                    response_size,
                    _SLIDING_WINDOW,
                    _TTL_SECONDS,
                ],
            )
        except Exception as e:
            logger.error(f"Redis error for attack features ({user_id}): {e}")
            return None

        # Parse the structured response
        total = int(raw[0])
        peak_rps = int(raw[1])
        sensitive_count = int(raw[2])
        header_score_val = int(raw[3])

        # Split by markers
        raw_str = raw
        ua_start = raw_str.index("|||UA|||") + 1
        resp_start = raw_str.index("|||RESP|||") + 1
        vel_start = raw_str.index("|||VEL|||") + 1

        ua_end = raw_str.index("|||RESP|||")
        resp_end = raw_str.index("|||VEL|||")

        ua_pairs = raw_str[ua_start:ua_end]
        resp_sizes_raw = raw_str[resp_start:vel_start - 1]
        vel_pairs = raw_str[vel_start:]

        if total < 3:
            return None

        ua_freq = {}
        for i in range(0, len(ua_pairs), 2):
            ua_freq[ua_pairs[i]] = int(ua_pairs[i + 1])

        resp_sizes = [float(v) for v in resp_sizes_raw]

        vel_buckets = {}
        for i in range(0, len(vel_pairs), 2):
            vel_buckets[int(vel_pairs[i])] = int(vel_pairs[i + 1])

        features = {
            "SAGE_Peak_Burst_RPS": float(peak_rps),
            "SAGE_Velocity_Trend": self._velocity_slope(vel_buckets),
            "SAGE_Sensitive_Endpoint_Ratio": sensitive_count / total,
            "SAGE_UA_Entropy": self._shannon_entropy(ua_freq),
            "SAGE_Header_Completeness": header_score_val / len(_BROWSER_HEADERS),
            "SAGE_Response_Size_Variance": self._std_dev(resp_sizes),
        }

        logger.debug(
            f"[ATTACK] User {user_id} | "
            f"PeakRPS={features['SAGE_Peak_Burst_RPS']:.0f} "
            f"VelTrend={features['SAGE_Velocity_Trend']:.4f} "
            f"SensitiveR={features['SAGE_Sensitive_Endpoint_Ratio']:.4f} "
            f"UA_Ent={features['SAGE_UA_Entropy']:.4f} "
            f"HeaderC={features['SAGE_Header_Completeness']:.2f} "
            f"RespVar={features['SAGE_Response_Size_Variance']:.2f}"
        )
        return features

    # ------------------------------------------------------------------
    # Feature math
    # ------------------------------------------------------------------

    @staticmethod
    def _velocity_slope(buckets: dict[int, int]) -> float:
        """
        Linear regression slope of per-minute request counts.
        Positive slope = accelerating attack (recon → exploit).
        Zero slope = sustained (flood).
        """
        if len(buckets) < 2:
            return 0.0

        sorted_mins = sorted(buckets.keys())
        n = len(sorted_mins)
        xs = list(range(n))
        ys = [buckets[m] for m in sorted_mins]

        x_mean = sum(xs) / n
        y_mean = sum(ys) / n

        numerator = sum((xs[i] - x_mean) * (ys[i] - y_mean) for i in range(n))
        denominator = sum((xs[i] - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return 0.0
        return numerator / denominator

    @staticmethod
    def _shannon_entropy(freq: dict[str, int]) -> float:
        """Shannon entropy over a frequency distribution."""
        total = sum(freq.values())
        if total == 0:
            return 0.0
        entropy = 0.0
        for count in freq.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        return entropy

    @staticmethod
    def _score_headers(headers: dict[str, str]) -> int:
        """Count how many standard browser headers are present (0-4)."""
        if not headers:
            return 0
        score = 0
        for h in _BROWSER_HEADERS:
            if h.lower() in {k.lower() for k in headers}:
                score += 1
        return score

    @staticmethod
    def _is_sensitive(path: str) -> bool:
        """Check if the endpoint is a known sensitive/admin path."""
        path_lower = path.lower().split("?")[0]
        for pattern in _SENSITIVE_PATTERNS:
            if path_lower.startswith(pattern) or path_lower == pattern:
                return True
        return False

    @staticmethod
    def _std_dev(values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        n = len(values)
        mean = sum(values) / n
        variance = sum((x - mean) ** 2 for x in values) / n
        return math.sqrt(variance)
