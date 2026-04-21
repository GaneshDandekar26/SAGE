package com.sage.gateway.filter;

import com.sage.gateway.filter.GatewayFilter;
import com.sage.gateway.service.RateLimiterService;
import jakarta.servlet.http.HttpServletRequest;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Component;

import java.util.Map;
import java.util.Optional;

@Component
public class RateLimitFilter implements GatewayFilter {

    private final RateLimiterService rateLimiterService;

    public RateLimitFilter(RateLimiterService rateLimiterService) {
        this.rateLimiterService = rateLimiterService;
    }

    @Override
    public String getName() {
        return "RateLimit";
    }

    @Override
    public Optional<ResponseEntity<String>> filter(HttpServletRequest request, Map<String, String> config) {
        System.out.println("🛑 RATE LIMITER ACTIVATED: Checking tokens for " + request.getRequestURI());

        String ipAddress = request.getHeader("X-Forwarded-For");
        if (ipAddress == null || ipAddress.isEmpty()) {
            ipAddress = request.getRemoteAddr();
        }
        String routeId = config.getOrDefault("routeId", "default-route");
        // Execute Redis Lua Script
        boolean isAllowed = rateLimiterService.isAllowed(ipAddress, routeId);

        System.out.println("RateLimit config = " + config);
        System.out.println("Resolved routeId = " + routeId);

        if (!isAllowed) {
            // Return the 429 response wrapped in an Optional
            return Optional.of(
                    ResponseEntity.status(HttpStatus.TOO_MANY_REQUESTS)
                            .body("{\"error\": \"Too Many Requests\"}")
            );
        }

        // SUCCESS: Return an empty Optional so the pipeline continues
        return Optional.empty();
    }
}