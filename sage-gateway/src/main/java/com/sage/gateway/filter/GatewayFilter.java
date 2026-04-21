package com.sage.gateway.filter;

import jakarta.servlet.http.HttpServletRequest;
import org.springframework.http.ResponseEntity;

import java.util.Map;
import java.util.Optional;

public interface GatewayFilter {

    /**
     * @return Optional.of(ResponseEntity) if the request is BLOCKED.
     * Optional.empty() if the request is ALLOWED to proceed.
     */
    Optional<ResponseEntity<String>> filter(HttpServletRequest request, Map<String, String> config);

    String getName();
}
