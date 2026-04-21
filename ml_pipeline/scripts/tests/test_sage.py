import requests
import time

GATEWAY_URL = "http://localhost:8081/api/get"

def test_tier1_hammer():
    print("--- 🔨 Testing Tier 1: Rate Limiting ---")
    print(f"Targeting: {GATEWAY_URL}")
    print("Goal: Drain the bucket (Burst: 2) and trigger a 429.\n")

    found_429 = False

    for i in range(1, 7):
        try:
            # We use a short timeout to keep the test snappy
            response = requests.get(GATEWAY_URL, timeout=3, headers={"Accept-Encoding": "identity"})

            status = response.status_code
            print(f"Request {i}: Status {status}")

            if status == 429:
                print("\n✅ SUCCESS: SAGE intercepted the burst and returned 429!")
                found_429 = True
                break

        except requests.exceptions.ConnectionError:
            print(f"❌ ERROR: Connection Refused. Is SAGE Gateway actually running on 8081?")
            return

    if not found_429:
        print("\n❌ FAIL: Sent 6 requests but never received a 429.")
        print("Check: Is Redis running? Is the 'RateLimit' filter attached to this route?")

if __name__ == "__main__":
    test_tier1_hammer()
