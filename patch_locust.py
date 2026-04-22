import re

with open("load-tests/locustfile.py", "r") as f:
    code = f.read()

# Human
code = re.sub(
    r'class HumanBrowser\(HttpUser\):(.*?)def _headers\(self\):.*?return \{\n\s*"X-Forwarded-For": ip_residential\(\)',
    r'class HumanBrowser(HttpUser):\g<1>def on_start(self):\n        self.ip = ip_residential()\n        headers = self._headers()\n        self.client.get("/static/style.css", headers=headers, name="Human - static css")\n        self.client.get("/static/app.js", headers=headers, name="Human - static js")\n        self.client.get("/static/logo.png", headers=headers, name="Human - static logo")\n\n    def _headers(self):\n        return {\n            "X-Forwarded-For": self.ip',
    code, flags=re.DOTALL
)

# Akamai
code = re.sub(
    r'class AkamaiScraper\(HttpUser\):(.*?)def on_start\(self\):\n\s*self\._id_cycle = cycle\(range\(1, 51\)\)\n\n\s*def _headers\(self\):.*?return \{\n\s*"X-Forwarded-For": ip_datacenter\(\)',
    r'class AkamaiScraper(HttpUser):\g<1>def on_start(self):\n        self.ip = ip_datacenter()\n        self._id_cycle = cycle(range(1, 51))\n\n    def _headers(self):\n        return {\n            "X-Forwarded-For": self.ip',
    code, flags=re.DOTALL
)

# Cloudflare Flood
code = re.sub(
    r'class CloudflareFlood\(HttpUser\):(.*?)def _headers\(self\):.*?return \{\n\s*"X-Forwarded-For": ip_distributed\(\)',
    r'class CloudflareFlood(HttpUser):\g<1>def on_start(self):\n        self.ip = ip_distributed()\n\n    def _headers(self):\n        return {\n            "X-Forwarded-For": self.ip',
    code, flags=re.DOTALL
)

# Unprotected Flood
code = re.sub(
    r'class UnprotectedFlood\(HttpUser\):(.*?)def _headers\(self\):.*?return \{\n\s*"X-Forwarded-For": ip_distributed\(\)',
    r'class UnprotectedFlood(HttpUser):\g<1>def on_start(self):\n        self.ip = ip_distributed()\n\n    def _headers(self):\n        return {\n            "X-Forwarded-For": self.ip',
    code, flags=re.DOTALL
)

# Recon
code = re.sub(
    r'class ReconBot\(HttpUser\):(.*?)def _headers\(self\):.*?return \{\n\s*"X-Forwarded-For": ip_tor_like\(\)',
    r'class ReconBot(HttpUser):\g<1>def on_start(self):\n        self.ip = ip_tor_like()\n\n    def _headers(self):\n        return {\n            "X-Forwarded-For": self.ip',
    code, flags=re.DOTALL
)

# Cleanup the old on_start for human
code = re.sub(r'def on_start\(self\):\n\s*headers = self._headers\(\)\n\s*self.client.get\("/static/style.css", headers=headers, name="Human - static css"\)\n\s*self.client.get\("/static/app.js", headers=headers, name="Human - static js"\)\n\s*self.client.get\("/static/logo.png", headers=headers, name="Human - static logo"\)', '', code)

with open("load-tests/locustfile.py", "w") as f:
    f.write(code)
