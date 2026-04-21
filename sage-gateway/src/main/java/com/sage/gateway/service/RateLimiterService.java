package com.sage.gateway.service;

import com.sage.gateway.config.RateLimitProperties;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.data.redis.core.script.RedisScript;
import org.springframework.stereotype.Service;

import java.time.Instant;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

@Service
public class RateLimiterService {

    private static final Logger logger = LoggerFactory.getLogger(RateLimiterService.class);

    private final StringRedisTemplate redisTemplate;
    private final RedisScript<Long> rateLimitScript;
    private final RateLimitProperties rateLimitProperties;


    public RateLimiterService(StringRedisTemplate redisTemplate, RedisScript<Long> rateLimitScript, RateLimitProperties rateLimitProperties) {
        this.redisTemplate = redisTemplate;
        this.rateLimitScript = rateLimitScript;
        this.rateLimitProperties = rateLimitProperties;
    }


    public boolean isAllowed(String ipAddress, String routeId) {

        // Resolve rate limits based on the Route being accessed
        RateLimitProperties.Policy policy = rateLimitProperties.getRoutes()
                .getOrDefault(routeId, rateLimitProperties.getDefaultPolicy());

        // Build a single, highly isolated Redis key
        // Format: sage:ratelimit:ip:{ip}:route:{route}
        // Ex : sage:ratelimit:ip:192.168.1.5:route:auth-route
        String routeKey = "sage:ratelimit:ip:" + ipAddress + ":route:" + routeId;

        List<String> keys = Collections.singletonList(routeKey);

        long now = Instant.now().getEpochSecond();
        Object[] args = {
                String.valueOf(policy.getReplenishRate()), // ARGV[1]: Route Limit
                String.valueOf(policy.getBurstCapacity()), // ARGV[2]: Route Burst
                String.valueOf(now),                       // ARGV[3]: Timestamp
                "1"                                        // ARGV[4]: Cost (1 token)
        };

        //  Execute Script
        try {
            Long result = redisTemplate.execute(rateLimitScript, keys, args);
            return result != null && result == 1L;
        } catch (Exception e) {
            // FAIL OPEN STRATEGY:
            // If Redis crashes, we let traffic through to the ML pipeline.
            logger.error("Redis Rate Limiter failed: {}", e.getMessage());
            return true;
        }
    }

    // Helper methods used as configuration placeholders
    // (to be replaced with actual configuration or lookup logic).

    private int[] getUserLimits(String userId) {
        // Returns {rate, burst} values.
        // Higher limits are assigned for VIP users.
        if ("vip_user".equals(userId)) return new int[] { 500, 1000 };
        return new int[] { 1, 5 }; // Default User
    }

    private int[] getTenantLimits(String tenantId) {
        // Partner A vs Partner B
        if ("partner_b".equals(tenantId)) return new int[] { 20_000, 40_000 };
        return new int[] { 10_000, 20_000 }; // Default Partner
    }
}