# ml_pipeline/features/request_velocity.py
import redis
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class RequestVelocityCalculator:
    def __init__(self, redis_host='localhost', redis_port=6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)

        # Lua script for a sliding 60-second time window using a Sorted Set
        self.lua_script = self.redis_client.register_script("""
            local key = KEYS[1]
            local current_ts_ms = tonumber(ARGV[1])
            local window_ms = tonumber(ARGV[2])
            local request_id = ARGV[3]

            -- Calculate the cutoff time (e.g., exactly 60 seconds ago)
            local window_start = current_ts_ms - window_ms

            -- 1. Prune the tree: Remove all requests older than the cutoff
            redis.call('ZREMRANGEBYSCORE', key, '-inf', window_start)

            -- 2. Add the current request.
            -- Score = timestamp. Member = unique ID so requests at the exact same ms don't overwrite each other.
            redis.call('ZADD', key, current_ts_ms, request_id)

            -- 3. Count how many requests survived the pruning
            local velocity = redis.call('ZCARD', key)

            -- 4. Set TTL so if the user leaves, we don't leak memory
            redis.call('PEXPIRE', key, window_ms)

            return velocity
        """)

    def extract_ms_from_iso(self, timestamp_val):
        """Converts timestamp to raw millisecond integers (copied from our Variance calculator)."""
        try:
            if isinstance(timestamp_val, (int, float)):
                return int(timestamp_val)
            if isinstance(timestamp_val, str):
                clean_string = timestamp_val.replace('Z', '+00:00')
                dt = datetime.fromisoformat(clean_string)
                return int(dt.timestamp() * 1000)
        except Exception as e:
            return None

    def calculate(self, user_id, timestamp_val, request_id):
        current_ts_ms = self.extract_ms_from_iso(timestamp_val)
        if not current_ts_ms or not request_id:
            return None

        redis_key = f"features:velocity:{user_id}"

        try:
            # 60000 ms = 60 seconds sliding window
            velocity = self.lua_script(keys=[redis_key], args=[current_ts_ms, 60000, request_id])

            print(f"[MATH] User {user_id} | Velocity (req/min): {velocity}")
            return int(velocity)

        except Exception as e:
            logger.error(f"Redis error calculating velocity for {user_id}: {e}")
            return None