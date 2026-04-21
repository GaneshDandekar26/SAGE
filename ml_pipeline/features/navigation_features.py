# ml_pipeline/features/navigation_features.py
"""
Stage 1 – Navigation Pattern Features (Human vs Bot)

These features capture *how* users move through the site, identifying
the natural exploration, backtracking, and hierarchical navigation that
humans exhibit versus the systematic crawl patterns of bots.

Features:
    1. SAGE_Backtrack_Ratio       – Revisits to already-seen endpoints
    2. SAGE_Path_Entropy          – Shannon entropy of endpoint distribution
    3. SAGE_Referral_Chain_Depth  – Average depth of hierarchical navigation chains
"""

import math
import re
import redis
import logging

logger = logging.getLogger(__name__)

_SLIDING_WINDOW = 50    # last N endpoints to consider
_TTL_SECONDS = 86400    # 24 h


class NavigationFeaturesCalculator:
    """
    Tracks the sequence of normalized endpoints a user visits and derives
    navigation-pattern features that separate human browsing from bot crawling.
    """

    def __init__(self, redis_host="localhost", redis_port=6379):
        self.redis_client = redis.Redis(
            host=redis_host, port=redis_port, decode_responses=True
        )

        # Lua script: maintain an ordered endpoint history in a Redis List
        # and a frequency hash for fast entropy calculation.
        self.lua_script = self.redis_client.register_script("""
            local key_prefix  = KEYS[1]
            local endpoint    = ARGV[1]
            local window_size = tonumber(ARGV[2])
            local ttl         = tonumber(ARGV[3])

            local history_key = key_prefix .. ':history'
            local freq_key    = key_prefix .. ':freq'

            -- 1. Push new endpoint to the front of the history list
            redis.call('LPUSH', history_key, endpoint)

            -- 2. If list exceeds window, pop the oldest and decrement its freq
            local list_len = redis.call('LLEN', history_key)
            if list_len > window_size then
                local evicted = redis.call('RPOP', history_key)
                if evicted then
                    local new_count = redis.call('HINCRBY', freq_key, evicted, -1)
                    if new_count <= 0 then
                        redis.call('HDEL', freq_key, evicted)
                    end
                end
            end

            -- 3. Increment frequency of the new endpoint
            redis.call('HINCRBY', freq_key, endpoint, 1)

            -- 4. Refresh TTLs
            redis.call('EXPIRE', history_key, ttl)
            redis.call('EXPIRE', freq_key, ttl)

            -- 5. Return the full history (most recent first) and freq map
            local history = redis.call('LRANGE', history_key, 0, -1)
            local freq_pairs = redis.call('HGETALL', freq_key)

            -- Encode freq as interleaved key/value in a second table
            -- Return both tables concatenated with a separator
            local result = {}
            for i, v in ipairs(history) do
                table.insert(result, v)
            end
            table.insert(result, '|||')  -- separator
            for i, v in ipairs(freq_pairs) do
                table.insert(result, v)
            end
            return result
        """)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate(self, user_id: str, raw_path: str) -> dict | None:
        """
        Call on every incoming request with the raw URI path.
        Returns a dict with navigation features, or None if not enough data.
        """
        if not self._is_trackable(raw_path):
            return None

        normalized = self._normalize_path(raw_path)
        redis_key = f"features:nav:{user_id}"

        try:
            raw = self.lua_script(
                keys=[redis_key],
                args=[normalized, _SLIDING_WINDOW, _TTL_SECONDS],
            )
        except Exception as e:
            logger.error(f"Redis error for navigation features ({user_id}): {e}")
            return None

        # Parse the concatenated result
        separator_idx = raw.index("|||")
        history = raw[:separator_idx]
        freq_pairs = raw[separator_idx + 1:]

        if len(history) < 3:
            return None  # Need at least 3 page visits

        # Build frequency dict
        freq = {}
        for i in range(0, len(freq_pairs), 2):
            freq[freq_pairs[i]] = int(freq_pairs[i + 1])

        features = {
            "SAGE_Backtrack_Ratio": self._backtrack_ratio(history),
            "SAGE_Path_Entropy": self._path_entropy(freq, len(history)),
            "SAGE_Referral_Chain_Depth": self._referral_chain_depth(history),
        }

        logger.debug(
            f"[NAV] User {user_id} | "
            f"Backtrack={features['SAGE_Backtrack_Ratio']:.4f} "
            f"PathEnt={features['SAGE_Path_Entropy']:.4f} "
            f"ChainD={features['SAGE_Referral_Chain_Depth']:.4f}"
        )
        return features

    # ------------------------------------------------------------------
    # Feature math
    # ------------------------------------------------------------------

    @staticmethod
    def _backtrack_ratio(history: list[str]) -> float:
        """
        Fraction of requests that revisit a previously-visited endpoint.
        Humans go back to compare pages; bots follow forward-only paths.
        """
        seen = set()
        backtracks = 0
        for endpoint in reversed(history):  # history is most-recent-first
            if endpoint in seen:
                backtracks += 1
            seen.add(endpoint)
        return backtracks / len(history) if history else 0.0

    @staticmethod
    def _path_entropy(freq: dict[str, int], total: int) -> float:
        """
        Shannon entropy of the endpoint frequency distribution.
        High entropy = diverse human browsing.  Low = repetitive bot crawl.
        """
        if total == 0:
            return 0.0
        entropy = 0.0
        for count in freq.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        return entropy

    @staticmethod
    def _referral_chain_depth(history: list[str]) -> float:
        """
        Measures hierarchical navigation chains.
        e.g., /products → /products/123 → /products/123/reviews = depth 3.
        Humans navigate in meaningful chains; bots jump randomly or enumerate.
        """
        if len(history) < 2:
            return 0.0

        chains = []
        current_chain = 1

        # Walk history in chronological order (reversed since list is newest-first)
        ordered = list(reversed(history))
        for i in range(1, len(ordered)):
            prev = ordered[i - 1]
            curr = ordered[i]
            # Check if current path extends the previous (hierarchical navigation)
            if curr.startswith(prev.rstrip("/") + "/") or prev.startswith(curr.rstrip("/") + "/"):
                current_chain += 1
            else:
                if current_chain > 1:
                    chains.append(current_chain)
                current_chain = 1

        if current_chain > 1:
            chains.append(current_chain)

        return sum(chains) / len(chains) if chains else 1.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_trackable(path: str) -> bool:
        """Filter out internal/health endpoints."""
        if path.startswith("/actuator") or path.startswith("/health"):
            return False
        return True

    @staticmethod
    def _normalize_path(path: str) -> str:
        """Strip query params and replace variable segments with wildcards."""
        path = path.split("?")[0]
        # Replace UUIDs
        path = re.sub(
            r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
            "*",
            path,
        )
        # Replace numeric IDs
        path = re.sub(r"/\d+", "/*", path)
        return path
