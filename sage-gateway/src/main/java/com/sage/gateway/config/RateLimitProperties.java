package com.sage.gateway.config;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Configuration;
import java.util.HashMap;
import java.util.Map;

@Configuration
@ConfigurationProperties(prefix = "sage.rate-limit")
public class RateLimitProperties {

    private Policy defaultPolicy = new Policy(100, 200); // Safe fallback
    private Map<String, Policy> routes = new HashMap<>();

    // Getters and Setters are required for Spring to inject the YAML values
    public Policy getDefaultPolicy() { return defaultPolicy; }
    public void setDefaultPolicy(Policy defaultPolicy) { this.defaultPolicy = defaultPolicy; }
    public Map<String, Policy> getRoutes() { return routes; }
    public void setRoutes(Map<String, Policy> routes) { this.routes = routes; }

    // Inner class representing the bucket config
    public static class Policy {
        private int replenishRate;
        private int burstCapacity;

        public Policy() {} // Default constructor for Spring
        public Policy(int replenishRate, int burstCapacity) {
            this.replenishRate = replenishRate;
            this.burstCapacity = burstCapacity;
        }
        // Getters and Setters
        public int getReplenishRate() { return replenishRate; }
        public void setReplenishRate(int replenishRate) { this.replenishRate = replenishRate; }
        public int getBurstCapacity() { return burstCapacity; }
        public void setBurstCapacity(int burstCapacity) { this.burstCapacity = burstCapacity; }
    }
}