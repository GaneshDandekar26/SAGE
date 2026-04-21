import requests
import time
import concurrent.futures

# Replace with your actual Gateway URL
GATEWAY_URL = "http://localhost:8081/api/test-route"

def fire_request(req_id):
    try:
        start = time.perf_counter()
        response = requests.get(GATEWAY_URL)
        latency = (time.perf_counter() - start) * 1000

        if response.status_code == 403:
            print(f"🚨 [BLOCKED] Req {req_id} | SAGE Firewall Dropped Connection! ({latency:.2f}ms)")
            return True # Bot detected
        else:
            print(f"✅ [ALLOWED] Req {req_id} | Status: {response.status_code} ({latency:.2f}ms)")
            return False

    except requests.exceptions.RequestException as e:
        print(f"Connection failed: {e}")
        return False

print("🚀 Launching Multi-Threaded Bot Attack against SAGE Engine...")

# Fire 200 requests across 10 concurrent threads to simulate a real L7 Flood
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    results = list(executor.map(fire_request, range(200)))

blocked_count = sum(results)
print(f"\n🏁 Attack complete. SAGE Engine blocked {blocked_count} out of 200 requests.")