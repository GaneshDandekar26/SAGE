import requests
import time
import random

API_URL = "http://localhost:8000/predict/"

print("Starting SAGE Engine Stress Test. Press Ctrl+C to stop.")
try:
    while True:
        # Simulate random user ID
        user_id = f"user_{random.randint(1, 1000)}"

        # Fire the request
        requests.get(f"{API_URL}{user_id}")

        # Sleep for a tiny fraction of a second to simulate continuous traffic
        # (e.g. 20-50 requests per second)
        time.sleep(random.uniform(0.01, 0.05))

except KeyboardInterrupt:
    print("\nStress test stopped.")