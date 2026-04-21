import requests
import time
import threading
import sys
import random

# Protected route configured in the Java gateway
TARGET_URL = "http://localhost:8081/api/get"

def print_status(profile, req_num, response):
    status = response.status_code
    if status == 200:
        print(f"[{profile}] Req {req_num}: 🟢 200 OK (Allowed)")
    elif status == 403:
        print(f"[{profile}] Req {req_num}: 🔴 403 FORBIDDEN (SAGE ML Ban!)")
    elif status == 429:
        print(f"[{profile}] Req {req_num}: 🟡 429 TOO MANY REQUESTS (Tier 1 Rate Limit)")
    else:
        print(f"[{profile}] Req {req_num}: ⚪ {status}")

def simulate_human():
    print("\n" + "="*50)
    print("🚶 SIMULATING: Normal Human Traffic (Benign)")
    print("="*50)

    # Humans browse different pages (creates Behavioral Diversity)
    endpoints = [
        "?page=home",
        "?page=products&id=12",
        "?page=about_us",
        "?page=contact",
        "?page=checkout"
    ]

    for i in range(1, 6):
        try:
            # Pick a random "page" to visit
            url = TARGET_URL + random.choice(endpoints)
            res = requests.get(url)
            print_status("Human", i, res)

            # Humans pause for random amounts of time to read
            human_reading_time = random.uniform(1.5, 4.0)
            time.sleep(human_reading_time)
        except Exception as e:
            print(f"Connection failed: {e}")

def simulate_scraper():
    print("\n" + "="*50)
    print("🤖 SIMULATING: Aggressive Web Scraper (Bot)")
    print("="*50)
    for i in range(1, 25):
        try:
            res = requests.get(TARGET_URL)
            print_status("Scraper", i, res)
            time.sleep(0.1) # Bots don't read
        except Exception as e:
            print(f"Connection failed: {e}")
            break

def attack_thread(thread_id):
    for i in range(20):
        try:
            res = requests.get(TARGET_URL)
            if res.status_code in [403, 429]:
                print(f"[Flood Thread {thread_id}] 🛡️ Blocked! Status: {res.status_code}")
                break
        except:
            pass

def simulate_flood():
    print("\n" + "="*50)
    print("🌊 SIMULATING: HTTP Flood (DDoS)")
    print("="*50)
    threads = []
    # Launch 20 concurrent threads to hammer the gateway
    for i in range(20):
        t = threading.Thread(target=attack_thread, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()
    print("[Flood] Attack wave complete.")

if __name__ == "__main__":
    print("SAGE Engine - Attack Simulation Tool")
    print("1. Simulate Normal Human")
    print("2. Simulate Aggressive Scraper")
    print("3. Simulate HTTP Flood")

    choice = input("\nSelect attack profile (1-3): ")

    if choice == '1':
        simulate_human()
    elif choice == '2':
        simulate_scraper()
    elif choice == '3':
        simulate_flood()
    else:
        print("Invalid choice.")
