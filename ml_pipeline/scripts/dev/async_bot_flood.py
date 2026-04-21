import asyncio
import aiohttp
import time

GATEWAY_URL = "http://localhost:8081/api/test-route"
TOTAL_REQUESTS = 20

async def fire_request(session, req_id):
    try:
        start = time.perf_counter()
        # Fire the request and immediately yield control back to the event loop
        async with session.get(GATEWAY_URL) as response:
            status = response.status
            # Read the text to ensure the response fully completed
            await response.text()
            latency = (time.perf_counter() - start) * 1000

            if status == 403:
                print(f"🚨 [BLOCKED] Req {req_id:03} | SAGE Dropped Connection! ({latency:.2f}ms)")
                return True
            else:
                print(f"✅ [ALLOWED] Req {req_id:03} | Status: {status} ({latency:.2f}ms)")
                return False

    except Exception as e:
        # If the gateway panics and drops the TCP connection entirely
        print(f"💥 [FAILED]  Req {req_id:03} | Connection violently rejected: {type(e).__name__}")
        return True # We count a dropped connection as a successful block

async def main():
    print(f"🚀 Launching Asynchronous 'Nuclear' Bot Attack ({TOTAL_REQUESTS} concurrent reqs)...")

    # Override client-side connection limits so we can flood the server
    connector = aiohttp.TCPConnector(limit=TOTAL_REQUESTS)

    async with aiohttp.ClientSession(connector=connector) as session:
        # 1. Queue up all 200 missiles
        tasks = [fire_request(session, i) for i in range(TOTAL_REQUESTS)]

        # 2. Fire them ALL simultaneously
        results = await asyncio.gather(*tasks)

        blocked_count = sum(results)
        print(f"\n🏁 Attack complete. SAGE Engine blocked {blocked_count} out of {TOTAL_REQUESTS} requests.")

if __name__ == "__main__":
    # Windows-specific fix for an asyncio bug
    import sys
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main())