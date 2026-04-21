# ml_pipeline/features/endpoint_diversity.py
import re
import redis
import logging

logger = logging.getLogger(__name__)

class EndpointDiversityCalculator:
    def __init__(self, redis_host='localhost', redis_port=6379):
        # Initialize connection to your local Redis container
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)

        # We pre-load the Lua script into Redis memory.
        # This makes execution lightning fast because we only send the SHA hash over the network later, not the whole script.
        self.lua_script = self.redis_client.register_script("""
            local key_prefix = KEYS[1]
            local endpoint = ARGV[1]
            local ttl = tonumber(ARGV[2])

            local endpoints_key = key_prefix .. ':endpoints' -- A Redis SET
            local count_key = key_prefix .. ':count'         -- A Redis Integer

            -- 1. Add the normalized endpoint to the Set (duplicates are ignored naturally)
            redis.call('SADD', endpoints_key, endpoint)

            -- 2. Increment the total request counter
            local total_requests = redis.call('INCR', count_key)

            -- 3. Refresh the Time-To-Live (TTL) so old sessions expire automatically
            redis.call('EXPIRE', endpoints_key, ttl)
            redis.call('EXPIRE', count_key, ttl)

            -- 4. Calculate the diversity fraction
            local unique_endpoints = redis.call('SCARD', endpoints_key)
            local diversity = unique_endpoints / total_requests

            -- Return as a string to avoid Lua/Python float conversion issues
            return tostring(diversity)
        """)

    def is_valid_traffic(self, path):
        """The Bouncer: Ignore internal metrics, health checks, or static assets."""
        if path.startswith('/actuator') or path.startswith('/health'):
            return False
        # You could also filter out .css, .js, .png here if your gateway serves them
        return True

    def normalize_path(self, path):
        """The Translator: Strip variable data to find the true behavioral intent."""
        # 1. Strip query parameters (e.g., ?sort=asc)
        path = path.split('?')[0]

        # 2. Replace standard UUIDs with a wildcard
        # e.g., /user/123e4567-e89b-12d3-a456-426614174000 -> /user/*
        path = re.sub(r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}', '*', path)

        # 3. Replace numeric IDs with a wildcard
        # e.g., /api/products/992 -> /api/products/*
        path = re.sub(r'/\d+', '/*', path)

        return path

    def calculate(self, user_id, raw_path):
        """The main entry point called by the Kafka Consumer."""
        if not self.is_valid_traffic(raw_path):
            return None # Skip calculation

        normalized_path = self.normalize_path(raw_path)

        # Create a unique Redis key namespace for this user's diversity feature
        redis_key = f"features:diversity:{user_id}"

        try:
            # Execute the Lua script.
            # 86400 seconds = 24h TTL, as specified in your Phase 3 Strategic Guide.
            diversity_score = self.lua_script(keys=[redis_key], args=[normalized_path, 86400])

            # Print for our local testing visibility
            print(f"[MATH] User {user_id} | Path: {normalized_path} | Score: {float(diversity_score):.4f}")

            return float(diversity_score)

        except Exception as e:
            logger.error(f"Redis error calculating diversity for {user_id}: {e}")
            return None