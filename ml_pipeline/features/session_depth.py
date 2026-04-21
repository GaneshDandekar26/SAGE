# ml_pipeline/features/session_depth.py
import redis
import logging

logger = logging.getLogger(__name__)

class SessionDepthCalculator:
    def __init__(self, redis_host='localhost', redis_port=6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)

        # A hyper-fast Lua script to increment a counter and refresh its timeout
        self.lua_script = self.redis_client.register_script("""
            local key = KEYS[1]
            local ttl_seconds = tonumber(ARGV[1])

            -- Increment the depth counter
            local current_depth = redis.call('INCR', key)

            -- Refresh the expiration timer (e.g., 30 mins of inactivity kills the session)
            redis.call('EXPIRE', key, ttl_seconds)

            return current_depth
        """)

    def calculate(self, session_id):
        if not session_id:
            return None

        redis_key = f"features:depth:{session_id}"

        try:
            # 1800 seconds = 30 minute standard web session
            depth = self.lua_script(keys=[redis_key], args=[1800])

            print(f"[MATH] Session {session_id} | Depth: {depth}")
            return int(depth)

        except Exception as e:
            logger.error(f"Redis error calculating depth for {session_id}: {e}")
            return None