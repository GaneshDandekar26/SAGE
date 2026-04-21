# ml_pipeline/features/content_features.py
"""
Stage 1 – Content Interaction Features (Human vs Bot)

These features capture *what* users do with content — whether they load
stylesheets, submit forms, trigger errors, or send varied payloads.
Real browsers produce rich content fingerprints; bare-bones bots do not.

Features:
    1. SAGE_Method_Diversity   – Unique HTTP methods / total requests
    2. SAGE_Static_Asset_Ratio – Static resource loads / total requests
    3. SAGE_Error_Rate         – 4xx/5xx responses / total requests
    4. SAGE_Payload_Variance   – StdDev of request payload sizes
"""

import math
import redis
import logging

logger = logging.getLogger(__name__)

_TTL_SECONDS = 86400
_SLIDING_WINDOW = 50  # payload sizes to keep for variance

# File extensions that count as static assets
_STATIC_EXTENSIONS = frozenset([
    ".css", ".js", ".png", ".jpg", ".jpeg", ".gif", ".svg",
    ".woff", ".woff2", ".ttf", ".eot", ".ico", ".webp", ".map",
])


class ContentFeaturesCalculator:
    """
    Tracks content-interaction signals that separate real browser sessions
    from headless/scripted bot traffic.
    """

    def __init__(self, redis_host="localhost", redis_port=6379):
        self.redis_client = redis.Redis(
            host=redis_host, port=redis_port, decode_responses=True
        )

        # Lua script: update all content-feature counters atomically
        self.lua_script = self.redis_client.register_script("""
            local key_prefix   = KEYS[1]
            local http_method  = ARGV[1]
            local is_static    = tonumber(ARGV[2])   -- 1 or 0
            local is_error     = tonumber(ARGV[3])   -- 1 or 0
            local payload_size = tonumber(ARGV[4])
            local window_size  = tonumber(ARGV[5])
            local ttl          = tonumber(ARGV[6])

            local total_key       = key_prefix .. ':total'
            local methods_key     = key_prefix .. ':methods'
            local static_key      = key_prefix .. ':static'
            local error_key       = key_prefix .. ':errors'
            local payloads_key    = key_prefix .. ':payloads'

            -- 1. Increment total request counter
            local total = redis.call('INCR', total_key)
            redis.call('EXPIRE', total_key, ttl)

            -- 2. Track unique HTTP methods (SET)
            redis.call('SADD', methods_key, http_method)
            redis.call('EXPIRE', methods_key, ttl)
            local unique_methods = redis.call('SCARD', methods_key)

            -- 3. Increment static asset counter
            if is_static == 1 then
                redis.call('INCR', static_key)
            else
                -- ensure key exists for GET
                if redis.call('EXISTS', static_key) == 0 then
                    redis.call('SET', static_key, '0', 'EX', ttl)
                end
            end
            redis.call('EXPIRE', static_key, ttl)
            local static_count = tonumber(redis.call('GET', static_key) or '0')

            -- 4. Increment error counter
            if is_error == 1 then
                redis.call('INCR', error_key)
            else
                if redis.call('EXISTS', error_key) == 0 then
                    redis.call('SET', error_key, '0', 'EX', ttl)
                end
            end
            redis.call('EXPIRE', error_key, ttl)
            local error_count = tonumber(redis.call('GET', error_key) or '0')

            -- 5. Track payload sizes in a sliding window list
            redis.call('LPUSH', payloads_key, payload_size)
            redis.call('LTRIM', payloads_key, 0, window_size - 1)
            redis.call('EXPIRE', payloads_key, ttl)
            local payloads = redis.call('LRANGE', payloads_key, 0, -1)

            -- Return: total, unique_methods, static_count, error_count, then payload values
            local result = {
                tostring(total),
                tostring(unique_methods),
                tostring(static_count),
                tostring(error_count)
            }
            for _, v in ipairs(payloads) do
                table.insert(result, v)
            end
            return result
        """)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate(
        self,
        user_id: str,
        http_method: str,
        path: str,
        status_code: int,
        payload_size: float,
    ) -> dict | None:
        """
        Call on every request/response cycle.

        Args:
            user_id:      IP or session identifier
            http_method:  GET, POST, PUT, DELETE, etc.
            path:         Raw request URI
            status_code:  HTTP response status code
            payload_size: Request content-length in bytes (0 if none)

        Returns:
            Dict with the 4 content features, or None if too early.
        """
        is_static = 1 if self._is_static_asset(path) else 0
        is_error = 1 if status_code >= 400 else 0
        redis_key = f"features:content:{user_id}"

        try:
            raw = self.lua_script(
                keys=[redis_key],
                args=[
                    http_method.upper(),
                    is_static,
                    is_error,
                    payload_size,
                    _SLIDING_WINDOW,
                    _TTL_SECONDS,
                ],
            )
        except Exception as e:
            logger.error(f"Redis error for content features ({user_id}): {e}")
            return None

        total = int(raw[0])
        unique_methods = int(raw[1])
        static_count = int(raw[2])
        error_count = int(raw[3])
        payloads = [float(v) for v in raw[4:]]

        if total < 3:
            return None

        features = {
            "SAGE_Method_Diversity": unique_methods / total,
            "SAGE_Static_Asset_Ratio": static_count / total,
            "SAGE_Error_Rate": error_count / total,
            "SAGE_Payload_Variance": self._std_dev(payloads),
        }

        logger.debug(
            f"[CONTENT] User {user_id} | "
            f"MethodDiv={features['SAGE_Method_Diversity']:.4f} "
            f"StaticR={features['SAGE_Static_Asset_Ratio']:.4f} "
            f"ErrorR={features['SAGE_Error_Rate']:.4f} "
            f"PayloadVar={features['SAGE_Payload_Variance']:.2f}"
        )
        return features

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_static_asset(path: str) -> bool:
        """Check if the path points to a static resource."""
        clean = path.split("?")[0].lower()
        for ext in _STATIC_EXTENSIONS:
            if clean.endswith(ext):
                return True
        return False

    @staticmethod
    def _std_dev(values: list[float]) -> float:
        """Standard deviation of a list of floats."""
        if len(values) < 2:
            return 0.0
        n = len(values)
        mean = sum(values) / n
        variance = sum((x - mean) ** 2 for x in values) / n
        return math.sqrt(variance)
