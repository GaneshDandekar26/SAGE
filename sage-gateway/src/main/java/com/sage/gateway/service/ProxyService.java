package com.sage.gateway.service;

import io.github.resilience4j.circuitbreaker.annotation.CircuitBreaker;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;

import jakarta.servlet.http.HttpServletRequest;
import org.springframework.http.HttpMethod;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.HttpMediaTypeException;
import org.springframework.web.reactive.function.client.WebClient;

import java.util.Enumeration;
import java.util.Set;


@Service
public class ProxyService {

    private final WebClient webClient;

    public ProxyService(WebClient webClient){
        this.webClient = webClient;
    }
    /*
     * Core proxy execution flow:
     * - Builds the outbound request.
     * - Propagates headers from the incoming request.
     * - Forwards the request body (if present).
     * - Blocks for the downstream response (virtual-thread friendly).
     */

    private static final Set<String> RFC_HOP_BY_HOP = Set.of(
            "connection", "keep-alive", "proxy-authenticate",
            "proxy-authorization", "te", "trailers", "transfer-encoding", "upgrade"
    );

    @CircuitBreaker(name = "gatewayService", fallbackMethod = "forwardRequestFallback")
    public ResponseEntity<String> forwardRequest(String targetUrl, HttpMethod method, HttpServletRequest request, String body){
        var requestSpec = webClient.method(method).uri(targetUrl);

        Enumeration<String> headerNames = request.getHeaderNames();
        while (headerNames.hasMoreElements()) {
            String name = headerNames.nextElement();
            if (!RFC_HOP_BY_HOP.contains(name.toLowerCase()) && !name.equalsIgnoreCase("host")) {
                Enumeration<String> values = request.getHeaders(name);
                while (values.hasMoreElements()) {
                    requestSpec.header(name, values.nextElement());
                }
            }
        }
        // Sends the request, maps the response to ResponseEntity<String>,
        // and blocks for completion.
        return requestSpec.bodyValue(body != null ? body : "")
                .exchangeToMono(response -> response.toEntity(String.class))
                .map(entity -> {
                    HttpHeaders cleanHeaders = new HttpHeaders();
                    entity.getHeaders().forEach((name, values) -> {
                        // FIX 1: Pragmatic stripping of encoding/length because body is now decoded
                        if (!RFC_HOP_BY_HOP.contains(name.toLowerCase()) &&
                                !name.equalsIgnoreCase("content-encoding") &&
                                !name.equalsIgnoreCase("content-length")) {
                            cleanHeaders.addAll(name, values);
                        }
                    });
                    return new ResponseEntity<>(entity.getBody(), cleanHeaders, entity.getStatusCode());
                })
                .block();
    }
    public ResponseEntity<String> forwardRequestFallback(String targetUrl, HttpMethod method, HttpServletRequest request, String body, Throwable throwable) {
        System.out.println("🚨 CIRCUIT BREAKER TRIPPED! Target: " + targetUrl);
        System.out.println("🔥 Reason: " + throwable.getMessage());

        // Gracefully handles downstream failures by returning a controlled response
        // instead of propagating the exception to the client.
        String fallbackJson = "{\"error\": \"SAGE Gateway: Service Temporarily Unavailable.\", \"status\": 503}";

        return ResponseEntity
                .status(HttpStatus.SERVICE_UNAVAILABLE)
                .header("Content-Type", "application/json")
                .body(fallbackJson);
    }

}
