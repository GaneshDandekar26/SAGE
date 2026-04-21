import redis
import time

r = redis.Redis(host='localhost', port=6379, decode_responses=True)

def inject_user_data(user_id, diversity, variance, depth, velocity):
    # We use a Hash (HSET) because it's O(1) and keeps all 4 features
    # grouped under a single key for that specific user.
    key = f"user:features:{user_id}"
    data = {
        "endpoint_diversity": diversity,
        "temporal_variance": variance,
        "session_depth": depth,
        "request_velocity": velocity
    }
    r.hset(key, mapping=data)
    # Set a 24h TTL
    r.expire(key, 86400)
    print(f"Successfully injected data for {user_id}")

if __name__ == "__main__":
    # Simulate a "Normal User" (low velocity, low variance)
    inject_user_data("user_123", 0.2, 0.1, 5, 1.2)

    # Simulate a "Bot" (high velocity, high diversity - hitting many endpoints fast)
    inject_user_data("bot_999", 0.9, 0.05, 50, 45.0)