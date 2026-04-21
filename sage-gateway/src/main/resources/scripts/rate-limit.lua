
-- SAGE Engine: Route-Aware Token Bucket Rate Limiter


local route_key = KEYS[1]
local replenish_rate = tonumber(ARGV[1])
local burst_capacity = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local requested_tokens = tonumber(ARGV[4])

-- Safe TTL Calculation
local ttl = 60
if replenish_rate > 0 then
    local fill_time = math.ceil(burst_capacity / replenish_rate)
    ttl = math.max(fill_time * 2, 60)
end

-- Fetch State
local bucket = redis.call("HMGET", route_key, "tokens", "last_refill")
local current_tokens = tonumber(bucket[1])
local last_refill = tonumber(bucket[2])

if current_tokens == nil then
    current_tokens = burst_capacity
    last_refill = now
end

-- Calculate Token Delta
local delta_seconds = math.max(0, now - last_refill)
local tokens_to_add = delta_seconds * replenish_rate

current_tokens = math.min(burst_capacity, current_tokens + tokens_to_add)

-- Evaluate Request
local is_allowed = 0
if current_tokens >= requested_tokens then
    current_tokens = current_tokens - requested_tokens
    is_allowed = 1
end

-- Always advance the clock if time has passed, ensuring fractional tokens accumulate safely
if delta_seconds > 0 then
    last_refill = now
end

-- Save State
redis.call("HMSET", route_key, "tokens", current_tokens, "last_refill", last_refill)
redis.call("EXPIRE", route_key, ttl)

return is_allowed