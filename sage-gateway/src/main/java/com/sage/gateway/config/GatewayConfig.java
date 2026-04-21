package com.sage.gateway.config;

import com.sage.gateway.routing.RouteDefinition;
import com.sage.gateway.routing.RouteRegistry;
import org.springframework.beans.factory.annotation.Configurable;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.http.client.reactive.ReactorClientHttpConnector;
import reactor.netty.http.client.HttpClient;
import io.netty.resolver.DefaultAddressResolverGroup;

import java.util.Map;
import java.util.List;

@Configuration
public class GatewayConfig {


    @Bean
    public WebClient.Builder webClientBuilder() {
        return WebClient.builder();
    }

    @Bean
    public WebClient webClient(WebClient.Builder builder){
        /**
         * Creates the central WebClient instance for the Gateway.
         * * @param builder Auto-configured by Spring Boot. It comes with:
         * 1. JSON codecs (Jackson) for parsing request/response bodies.
         * 2. Micrometer hooks for tracking metrics (latency, errors).
         * 3. Standard timeout defaults.
         * @return A fully built WebClient ready to make high-concurrency requests.
         * We use this single instance throughout the app to reuse internal
         * resources (Connection Pool) rather than creating new clients per request.
         */
        HttpClient httpClient = HttpClient.create()
                .resolver(DefaultAddressResolverGroup.INSTANCE);

        return builder
                .clientConnector(new ReactorClientHttpConnector(httpClient))
                .build();
    }

    @Bean
    public RouteRegistry routeRegistry() {

        //basic route
        RouteDefinition publicUserRoute = new RouteDefinition(
                "public-user-route",
                "/api/public/{segment}",
                "https://jsonplaceholder.typicode.com",
                null // No filters
        );

        // secure route
        Map<String, Map<String, String>> secureFilters = Map.of(
                "RateLimit", Map.of(
                        "routeId", "secure-data-route", // Explicitly inject the route ID here
                        "replenishRate", "1",           // Optional now, since YAML handles limits,
                        "burstCapacity", "2"            // but safe to leave if filter still falls back to them
                )
        );

        RouteDefinition secureRoute = new RouteDefinition(
                "secure-data-route", // The actual ID
                "/api/get",
                "https://postman-echo.com",
                secureFilters
        );
        // Load them into the registry
        return new RouteRegistry(List.of(publicUserRoute, secureRoute));
    }
}
