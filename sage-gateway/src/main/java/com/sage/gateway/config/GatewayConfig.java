package com.sage.gateway.config;

import com.sage.gateway.routing.RouteDefinition;
import com.sage.gateway.routing.RouteRegistry;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.http.client.reactive.ReactorClientHttpConnector;
import org.springframework.beans.factory.annotation.Value;
import reactor.netty.http.client.HttpClient;
import io.netty.resolver.DefaultAddressResolverGroup;

import java.util.Map;
import java.util.List;

@Configuration
public class GatewayConfig {

    @Value("${UPSTREAM_URL:http://localhost:3001}")
    private String upstreamUrl;

    @Bean
    public WebClient.Builder webClientBuilder() {
        return WebClient.builder();
    }

    @Bean
    public WebClient webClient(WebClient.Builder builder){
        HttpClient httpClient = HttpClient.create()
                .resolver(DefaultAddressResolverGroup.INSTANCE);

        return builder
                .clientConnector(new ReactorClientHttpConnector(httpClient))
                .build();
    }

    @Bean
    public RouteRegistry routeRegistry() {
        return new RouteRegistry(List.of(
                new RouteDefinition("root", "/", upstreamUrl, null),
                new RouteDefinition("products", "/products", upstreamUrl, null),
                new RouteDefinition("products-id", "/products/{id}", upstreamUrl, null),
                new RouteDefinition("api-price", "/api/price/{id}", upstreamUrl, null),
                new RouteDefinition("api-inventory", "/api/inventory/{id}", upstreamUrl, null),
                new RouteDefinition("api-search", "/api/search", upstreamUrl, null),
                new RouteDefinition("static", "/static/{file}", upstreamUrl, null),
                new RouteDefinition("static-local", "/static/{dir}/{file}", upstreamUrl, null),
                new RouteDefinition("checkout", "/checkout", upstreamUrl, null),
                new RouteDefinition("cart", "/cart", upstreamUrl, null),
                new RouteDefinition("actuator", "/actuator/prometheus", upstreamUrl, null),
                new RouteDefinition("admin-routes", "/admin/routes", upstreamUrl, null),
                new RouteDefinition("echo", "/echo", upstreamUrl, null)
        ));
    }
}
