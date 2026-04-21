package com.sage.gateway.service;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;

import java.time.Instant;

@Service
public class TrafficLogger {

    private static final Logger log = LoggerFactory.getLogger("TRAFFIC_DATA");

    @Async
    public void logTraffic(String tenantId, String userId, String path, long durationMs, int statusCode){
        log.info("{{\"timestamp\": \"{}\", \"tenant\": \"{}\", \"user\": \"{}\", \"path\": \"{}\", \"latency_ms\": {}, \"status\": {}}}",
                Instant.now(), tenantId, userId, path, durationMs, statusCode);
    }
}
