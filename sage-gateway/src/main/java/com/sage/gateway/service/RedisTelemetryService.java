package com.sage.gateway.service;

import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Service;
import java.time.Instant;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

@Service
public class RedisTelemetryService {

    private final StringRedisTemplate redisTemplate;
    private static final int WINDOW_SIZE = 50;

    public RedisTelemetryService(StringRedisTemplate redisTemplate) {
        this.redisTemplate = redisTemplate;
    }

    public Map<String, Double> processAndGetTelemetry(String ip, double payloadSize, String path) {
        long currentTimestamp = Instant.now().toEpochMilli();
        String timeKey = "sage:timestamps:" + ip;
        String sizeKey = "sage:sizes:" + ip;
        String hashKey = "sage:telemetry:" + ip;

        // Request counters for e-commerce scraper-focused features.
        redisTemplate.opsForHash().increment(hashKey, "total_hits", 1);
        String safePath = path == null ? "" : path;
        if (safePath.matches("/api/price/.*|/api/inventory/.*|/products.*")) {
            redisTemplate.opsForHash().increment(hashKey, "product_hits", 1);
        }
        if ("/cart".equals(safePath) || "/checkout".equals(safePath)) {
            redisTemplate.opsForHash().increment(hashKey, "cart_hits", 1);
        }
        if (safePath.startsWith("/static/")) {
            redisTemplate.opsForHash().increment(hashKey, "static_hits", 1);
        }

        double sequentialTraversalScore = 0.0;
        if (safePath.matches("/products/\\d+|/api/price/\\d+|/api/inventory/\\d+")) {
            String resourceId = safePath.replaceAll(".*/", "");
            String seqKey = "sage:seq:" + ip;

            redisTemplate.opsForList().leftPush(seqKey, resourceId);
            redisTemplate.opsForList().trim(seqKey, 0, 19);
            redisTemplate.expire(seqKey, 5, TimeUnit.MINUTES);

            List<String> recentIds = redisTemplate.opsForList().range(seqKey, 0, -1);
            if (recentIds == null) {
                recentIds = new ArrayList<>();
            }

            sequentialTraversalScore = computeSequentialScore(
                    recentIds.stream().map(Object::toString).collect(Collectors.toList())
            );
        }

        // LPUSH new data to the front of the lists
        redisTemplate.opsForList().leftPush(timeKey, String.valueOf(currentTimestamp));
        redisTemplate.opsForList().leftPush(sizeKey, String.valueOf(payloadSize));

        // LTRIM to keep only the last WINDOW_SIZE elements
        redisTemplate.opsForList().trim(timeKey, 0, WINDOW_SIZE - 1);
        redisTemplate.opsForList().trim(sizeKey, 0, WINDOW_SIZE - 1);

        // Fetch the sliding window
        List<String> rawTimes = redisTemplate.opsForList().range(timeKey, 0, -1);
        List<String> rawSizes = redisTemplate.opsForList().range(sizeKey, 0, -1);

        // Parse to doubles
        List<Double> times = rawTimes.stream().map(Double::parseDouble).collect(Collectors.toList());
        List<Double> sizes = rawSizes.stream().map(Double::parseDouble).collect(Collectors.toList());

        // Calculate the 4 Core SAGE Features
        double sessionDepth = times.size();
        double requestVelocity = calculateVelocity(times);
        double temporalVariance = calculateTemporalVariance(times);
        double behavioralDiversity = calculateStandardDeviation(sizes);
        double totalHits = getCounter(hashKey, "total_hits");
        double productHits = getCounter(hashKey, "product_hits");
        double cartHits = getCounter(hashKey, "cart_hits");
        double staticHits = getCounter(hashKey, "static_hits");

        double endpointConcentration = totalHits > 0 ? productHits / totalHits : 0.0;
        double cartRatio = totalHits > 0 ? cartHits / totalHits : 0.0;
        double assetSkipRatio = totalHits > 0 ? 1.0 - (staticHits / totalHits) : 1.0;

        // Update the Redis Hash
        redisTemplate.opsForHash().put(hashKey, "SAGE_Session_Depth", String.valueOf(sessionDepth));
        redisTemplate.opsForHash().put(hashKey, "SAGE_Request_Velocity", String.valueOf(requestVelocity));
        redisTemplate.opsForHash().put(hashKey, "SAGE_Temporal_Variance", String.valueOf(temporalVariance));
        redisTemplate.opsForHash().put(hashKey, "SAGE_Behavioral_Diversity", String.valueOf(behavioralDiversity));
        redisTemplate.opsForHash().put(hashKey, "SAGE_Endpoint_Concentration", String.valueOf(endpointConcentration));
        redisTemplate.opsForHash().put(hashKey, "SAGE_Cart_Ratio", String.valueOf(cartRatio));
        redisTemplate.opsForHash().put(hashKey, "SAGE_Asset_Skip_Ratio", String.valueOf(assetSkipRatio));
        redisTemplate.opsForHash().put(hashKey, "SAGE_Sequential_Traversal", String.valueOf(sequentialTraversalScore));

        // Set a 1-hour TTL
        redisTemplate.expire(hashKey, Duration.ofHours(1));
        redisTemplate.expire(timeKey, Duration.ofHours(1));
        redisTemplate.expire(sizeKey, Duration.ofHours(1));

        return Map.of(
                "SAGE_Session_Depth", sessionDepth,
                "SAGE_Request_Velocity", requestVelocity,
                "SAGE_Temporal_Variance", temporalVariance,
                "SAGE_Behavioral_Diversity", behavioralDiversity,
                "SAGE_Endpoint_Concentration", endpointConcentration,
                "SAGE_Cart_Ratio", cartRatio,
                "SAGE_Asset_Skip_Ratio", assetSkipRatio,
                "SAGE_Sequential_Traversal", sequentialTraversalScore
        );
    }

    private double computeSequentialScore(List<String> ids) {
        if (ids == null || ids.size() < 3) {
            return 0.0;
        }

        List<String> deduped = new ArrayList<>();
        for (String id : ids) {
            if (deduped.isEmpty() || !id.equals(deduped.get(deduped.size() - 1))) {
                deduped.add(id);
            }
        }

        if (deduped.size() < 3) {
            return 0.0;
        }

        int consecutivePairs = 0;
        for (int i = 0; i < deduped.size() - 1; i++) {
            try {
                int current = Integer.parseInt(deduped.get(i));
                int next = Integer.parseInt(deduped.get(i + 1));
                // LPUSH order means index 0 is newest.
                if (next + 1 == current) {
                    consecutivePairs++;
                }
            } catch (NumberFormatException e) {
                // Ignore non-numeric IDs.
            }
        }

        return (double) consecutivePairs / (deduped.size() - 1);
    }

    private double getCounter(String hashKey, String field) {
        Object val = redisTemplate.opsForHash().get(hashKey, field);
        if (val == null) {
            return 0.0;
        }
        try {
            return Double.parseDouble(val.toString());
        } catch (NumberFormatException ex) {
            return 0.0;
        }
    }

    private double calculateVelocity(List<Double> times) {
        if (times.size() < 2) return 1.0;
        double timeSpanSeconds = (times.get(0) - times.get(times.size() - 1)) / 1000.0;

        if (timeSpanSeconds <= 0.001) timeSpanSeconds = 0.001;

        return times.size() / timeSpanSeconds;
    }

    private double calculateTemporalVariance(List<Double> times) {
        if (times.size() < 2) return 0.0;

        double[] iat = new double[times.size() - 1];
        double sumIat = 0;
        for (int i = 0; i < times.size() - 1; i++) {
            iat[i] = times.get(i) - times.get(i + 1);
            sumIat += iat[i];
        }

        double meanIat = sumIat / iat.length;
        if (meanIat == 0) return 0.0;

        double sumSquaredDiffs = 0;
        for (double val : iat) {
            sumSquaredDiffs += Math.pow(val - meanIat, 2);
        }
        double stdIat = Math.sqrt(sumSquaredDiffs / iat.length);

        return stdIat / meanIat;
    }

    private double calculateStandardDeviation(List<Double> values) {
        if (values.size() < 2) return 100.0;

        double sum = 0;
        for (double val : values) sum += val;
        double mean = sum / values.size();

        double sumSquaredDiffs = 0;
        for (double val : values) {
            sumSquaredDiffs += Math.pow(val - mean, 2);
        }
        return Math.sqrt(sumSquaredDiffs / values.size());
    }
    public boolean isIpBanned(String ip) {
        // Instantly checks if the IP is on the temporary ban list
        return Boolean.TRUE.equals(redisTemplate.hasKey("sage:ban:" + ip));
    }

    public void banIp(String ip) {
        // Bans the IP for 5 minutes (300 seconds)
        redisTemplate.opsForValue().set("sage:ban:" + ip, "true", java.time.Duration.ofMinutes(5));
    }

    public boolean isGlobalPathFlooding(String path, long thresholdPerSecond) {
        if (path == null || path.isBlank()) {
            return false;
        }

        long currentSecond = Instant.now().getEpochSecond();
        String safePath = path.replaceAll("[^a-zA-Z0-9_:/.-]", "_");
        String key = "sage:global:path:" + safePath + ":" + currentSecond;
        String blockKey = "sage:global:block:" + safePath;

        // If a short endpoint cooldown is active, keep blocking immediately.
        if (Boolean.TRUE.equals(redisTemplate.hasKey(blockKey))) {
            return true;
        }

        Long currentCount = redisTemplate.opsForValue().increment(key);
        if (currentCount != null && currentCount == 1L) {
            redisTemplate.expire(key, java.time.Duration.ofSeconds(2));
        }

        if (currentCount != null && currentCount > thresholdPerSecond) {
            // Create a short-lived global endpoint block to prevent per-second burst escapes.
            redisTemplate.opsForValue().set(blockKey, "1", java.time.Duration.ofSeconds(2));
            return true;
        }

        return false;
    }
}