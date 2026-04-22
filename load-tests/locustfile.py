import random
import sys
from itertools import cycle

from locust import HttpUser, between, task, tag


def ip_residential():
    if random.random() < 0.5:
        return f"192.168.{random.randint(0, 255)}.{random.randint(1, 254)}"
    return f"10.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"


def ip_datacenter():
    prefix = random.choice((52, 34))
    return f"{prefix}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"


def ip_distributed():
    return f"{random.randint(11, 223)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"


def ip_tor_like():
    prefix = random.choice((185, 176))
    return f"{prefix}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"


SCRAPER_UA_POOL = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.6312.124 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.6261.111 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.4; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:122.0) Gecko/20100101 Firefox/122.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.6367.82 Mobile Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edg/124.0.2478.80 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_2) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
]


FLOOD_SEARCH_TERMS = [
    "deal",
    "wireless",
    "gaming",
    "summer",
    "pro",
    "new",
    "sale",
    "premium",
    "eco",
    "smart",
]


SQLI_QUERIES = ["' OR 1=1", "../etc/passwd"]


@tag("human")
class HumanBrowser(HttpUser):
    scenario_tags = {"human"}
    fixed_count = 8
    wait_time = between(3, 8)

    def on_start(self):
        self.ip = ip_residential()
        headers = self._headers()
        self.client.get("/static/style.css", headers=headers, name="Human - static css")
        self.client.get("/static/app.js", headers=headers, name="Human - static js")
        self.client.get("/static/logo.png", headers=headers, name="Human - static logo")

    def _headers(self):
        return {
            "X-Forwarded-For": self.ip,
            "User-Agent": random.choice(SCRAPER_UA_POOL),
            "Content-Type": "application/json",
        }

    def _random_product_id(self):
        return random.randint(1, 50)

    

    @task(4)
    def browse_pages(self):
        headers = self._headers()
        path = random.choice(["/", "/products", "/cart"])
        self.client.get(path, headers=headers, name=f"Human - {path}")

    @task(3)
    def view_product_and_price(self):
        product_id = self._random_product_id()
        headers = self._headers()
        self.client.get(f"/products/{product_id}", headers=headers, name="Human - /products/:id")
        self.client.get(f"/api/price/{product_id}", headers=headers, name="Human - /api/price/:id")

        if random.random() < 0.30:
            self.client.post("/checkout", json={"status": "mock"}, headers=headers, name="Human - /checkout")


@tag("akamai_scraper")
class AkamaiScraper(HttpUser):
    scenario_tags = {"akamai_scraper"}
    fixed_count = 20
    wait_time = between(0.5, 1.5)

    def on_start(self):
        self.ip = ip_datacenter()
        self._id_cycle = cycle(range(1, 51))

    def _headers(self):
        return {
            "X-Forwarded-For": self.ip,
            "User-Agent": random.choice(SCRAPER_UA_POOL),
            "Content-Type": "application/json",
        }

    @task(1)
    def product_listing(self):
        self.client.get("/products", headers=self._headers(), name="Scraper - /products")

    @task(3)
    def price_inventory_sweep(self):
        product_id = next(self._id_cycle)
        headers = self._headers()
        self.client.get(f"/products/{product_id}", headers=headers, name="Scraper - /products/:id")
        self.client.get(f"/api/price/{product_id}", headers=headers, name="Scraper - /api/price/:id")
        self.client.get(f"/api/inventory/{product_id}", headers=headers, name="Scraper - /api/inventory/:id")


@tag("cloudflare_flood")
class CloudflareFlood(HttpUser):
    scenario_tags = {"cloudflare_flood"}
    fixed_count = 8
    wait_time = between(0, 0.1)

    def on_start(self):
        self.ip = ip_distributed()

    def _headers(self):
        return {
            "X-Forwarded-For": self.ip,
            "User-Agent": "curl/8.6.0",
            "Content-Type": "application/json",
        }

    @task(2)
    def flood_products(self):
        self.client.get("/products", headers=self._headers(), name="Flood - /products")

    @task(2)
    def flood_search(self):
        term = random.choice(FLOOD_SEARCH_TERMS)
        self.client.get(f"/api/search?q={term}", headers=self._headers(), name="Flood - /api/search")

    @task(3)
    def flood_price(self):
        product_id = random.randint(1, 50)
        self.client.get(f"/api/price/{product_id}", headers=self._headers(), name="Flood - /api/price/:id")


@tag("unprotected_flood")
class UnprotectedFlood(HttpUser):
    scenario_tags = {"unprotected_flood"}
    fixed_count = 8
    wait_time = between(0, 0.1)

    # Use absolute URLs so this class still targets 3001 even when global --host points to 8081.
    target_base = "http://localhost:3001"

    def on_start(self):
        self.ip = ip_distributed()

    def _headers(self):
        return {
            "X-Forwarded-For": self.ip,
            "User-Agent": "curl/8.6.0",
            "Content-Type": "application/json",
        }

    @task(2)
    def flood_products(self):
        self.client.get(f"{self.target_base}/products", headers=self._headers(), name="Unprotected Flood - /products")

    @task(2)
    def flood_search(self):
        term = random.choice(FLOOD_SEARCH_TERMS)
        self.client.get(
            f"{self.target_base}/api/search?q={term}",
            headers=self._headers(),
            name="Unprotected Flood - /api/search",
        )

    @task(3)
    def flood_price(self):
        product_id = random.randint(1, 50)
        self.client.get(
            f"{self.target_base}/api/price/{product_id}",
            headers=self._headers(),
            name="Unprotected Flood - /api/price/:id",
        )


@tag("recon")
class ReconBot(HttpUser):
    scenario_tags = {"recon"}
    fixed_count = 4
    wait_time = between(2, 5)

    def on_start(self):
        self.ip = ip_tor_like()

    def _headers(self):
        return {
            "X-Forwarded-For": self.ip,
            "User-Agent": "python-requests/2.31.0",
            "Content-Type": "application/json",
        }

    @task(2)
    def probe_admin_metrics(self):
        headers = self._headers()
        self.client.get("/actuator/prometheus", headers=headers, name="Recon - /actuator/prometheus")
        self.client.get("/admin/routes", headers=headers, name="Recon - /admin/routes")

    @task(2)
    def probe_search_injection(self):
        payload = random.choice(SQLI_QUERIES)
        self.client.get(f"/api/search?q={payload}", headers=self._headers(), name="Recon - /api/search (sqli)")

    @task(1)
    def probe_invalid_product(self):
        self.client.get("/products/99999", headers=self._headers(), name="Recon - /products/99999")

    @task(1)
    def probe_path_traversal(self):
        self.client.get("/static/../etc/passwd", headers=self._headers(), name="Recon - traversal")


def _parse_selected_tags(argv):
    selected = set()
    for idx, token in enumerate(argv):
        if token == "--tags" and idx + 1 < len(argv):
            selected.update(part.strip() for part in argv[idx + 1].split(",") if part.strip())
        elif token.startswith("--tags="):
            selected.update(part.strip() for part in token.split("=", 1)[1].split(",") if part.strip())
    return selected


def _apply_tag_based_user_activation():
    selected_tags = _parse_selected_tags(sys.argv)
    if not selected_tags:
        return

    user_classes = [
        HumanBrowser,
        AkamaiScraper,
        CloudflareFlood,
        UnprotectedFlood,
        ReconBot,
    ]

    for user_cls in user_classes:
        class_tags = getattr(user_cls, "scenario_tags", set())
        user_cls.abstract = class_tags.isdisjoint(selected_tags)


_apply_tag_based_user_activation()

# Run config
# locust -f locustfile.py --headless -u 40 -r 5 --run-time 5m --host http://localhost:3001
# HumanBrowser: 12 users
# AkamaiScraper: 14 users
# CloudflareFlood: 8 users
# UnprotectedFlood: 8 users
# ReconBot: 6 users