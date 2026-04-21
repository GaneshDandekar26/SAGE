# ml_pipeline/features/temporal_variance.py
import redis
import logging
import math
from datetime import datetime

logger = logging.getLogger(__name__)

class TemporalVarianceCalculator:
    def __init__(self, redis_host='localhost', redis_port=6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)

        # Lua script to maintain a sliding window of intervals and calculate CV
        self.lua_script = self.redis_client.register_script("""
            local key_prefix = KEYS[1]
            local current_ts_ms = tonumber(ARGV[1])
            local window_size = 20
            local ttl = 86400

            local last_ts_key = key_prefix .. ':last_ts'
            local intervals_key = key_prefix .. ':intervals'

            -- 1. Get the last timestamp
            local last_ts = redis.call('GET', last_ts_key)

            -- Update last timestamp for the NEXT request
            redis.call('SET', last_ts_key, current_ts_ms, 'EX', ttl)

            -- If this is their first request, we can't calculate a time difference yet
            if not last_ts then
                return "-1"
            end

            -- 2. Calculate the time difference (delta) in milliseconds
            local delta = math.abs(current_ts_ms - tonumber(last_ts))

            -- 3. Push delta to our sliding window list and trim to window_size
            redis.call('LPUSH', intervals_key, delta)
            redis.call('LTRIM', intervals_key, 0, window_size - 1)
            redis.call('EXPIRE', intervals_key, ttl)

            -- 4. Get all intervals in the current window to calculate math
            local intervals = redis.call('LRANGE', intervals_key, 0, -1)
            local count = #intervals

            if count < 3 then
                return "-1" -- Need at least 3 points for meaningful variance
            end

            -- 5. Calculate Mean
            local sum = 0
            for i=1, count do
                sum = sum + tonumber(intervals[i])
            end
            local mean = sum / count

            if mean == 0 then
                return "0.0" -- Prevent division by zero if all requests hit in the exact same millisecond
            end

            -- 6. Calculate Standard Deviation
            local sum_sq_diff = 0
            for i=1, count do
                local diff = tonumber(intervals[i]) - mean
                sum_sq_diff = sum_sq_diff + (diff * diff)
            end
            local variance = sum_sq_diff / count
            local std_dev = math.sqrt(variance)

            -- 7. Coefficient of Variation (CV = StdDev / Mean)
            local cv = std_dev / mean
            return tostring(cv)
        """)

    def extract_ms_from_iso(self, timestamp_val):
            """Converts timestamp to raw millisecond integers, handling both int and string formats."""
            try:
                # If the Java gateway is already sending epoch milliseconds as a number (Fast Path!)
                if isinstance(timestamp_val, (int, float)):
                    return int(timestamp_val)

                # Fallback just in case it ever sends an ISO string
                if isinstance(timestamp_val, str):
                    clean_string = timestamp_val.replace('Z', '+00:00')
                    dt = datetime.fromisoformat(clean_string)
                    return int(dt.timestamp() * 1000)

            except Exception as e:
                logger.error(f"Error parsing timestamp {timestamp_val}: {e}")
                return None

    def calculate(self, user_id, iso_timestamp):
        current_ts_ms = self.extract_ms_from_iso(iso_timestamp)
        if not current_ts_ms:
            return None

        redis_key = f"features:variance:{user_id}"

        try:
            cv_score = self.lua_script(keys=[redis_key], args=[current_ts_ms])

            if cv_score != "-1":
                print(f"[MATH] User {user_id} | Temporal CV: {float(cv_score):.4f}")
                return float(cv_score)
            return None

        except Exception as e:
            logger.error(f"Redis error calculating variance for {user_id}: {e}")
            return None