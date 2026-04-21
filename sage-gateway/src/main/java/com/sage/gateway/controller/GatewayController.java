package com.sage.gateway.controller;

import com.sage.gateway.filter.GatewayFilter;
import com.sage.gateway.routing.RouteDefinition;
import com.sage.gateway.routing.RouteRegistry;
import com.sage.gateway.routing.RouteResolver;
import com.sage.gateway.service.ProxyService;
import com.sage.gateway.service.TrafficLogger;
import jakarta.servlet.http.HttpServletRequest;
import org.springframework.http.HttpMethod;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.net.URISyntaxException;
import java.util.List;
import java.util.Map;
import java.util.Optional;

@RestController
public class GatewayController {

    private final ProxyService proxyService;
    private final TrafficLogger trafficLogger;
    private final RouteRegistry routeRegistry;
    private final RouteResolver routeResolver;
    private final List<GatewayFilter> availableFilters;

    public GatewayController(
            ProxyService proxyService,
            TrafficLogger trafficLogger,
            RouteResolver routeResolver,
            RouteRegistry routeRegistry,
            List<GatewayFilter> availableFilters) {
        this.proxyService = proxyService;
        this.trafficLogger = trafficLogger;
        this.routeResolver = routeResolver;
        this.routeRegistry = routeRegistry;
        this.availableFilters = availableFilters;
    }

    @GetMapping("/echo")
    public ResponseEntity<String> echo() {
        return ResponseEntity.ok("{\"status\":\"ok\",\"message\":\"SAGE echo\"}");
    }

    @RequestMapping("/**")
    public ResponseEntity<String> handleRequest(
            HttpServletRequest request,
            @RequestBody(required = false) String body,
            @RequestHeader(value = "X-Tenant-ID", defaultValue = "unknown") String tenantId,
            @RequestHeader(value = "X-User-ID", defaultValue = "anonymous") String userId)
            throws URISyntaxException {

        long startTime = System.currentTimeMillis();
        String path = request.getRequestURI();
        int statusCode = 500;

        try {
            RouteDefinition matchedRoute = routeResolver.resolve(path, request);

            if (matchedRoute == null) {
                statusCode = 404;
                return ResponseEntity.status(404).body("Gateway Error: Route Not Found");
            }

            Optional<ResponseEntity<String>> filterResult = applyPreProxyFilters(request, matchedRoute);
            if (filterResult.isPresent()) {
                statusCode = filterResult.get().getStatusCode().value();
                return filterResult.get();
            }

            String backendPath = path.replaceFirst("^/api", "");
            String target = matchedRoute.backendUrl() + backendPath;

            ResponseEntity<String> response = proxyService.forwardRequest(
                    target,
                    HttpMethod.valueOf(request.getMethod()),
                    request,
                    body
            );

            statusCode = response.getStatusCode().value();
            return response;

        } catch (Exception e) {
            statusCode = 500;
            e.printStackTrace();
            return ResponseEntity.internalServerError().body("Gateway Error: " + e.getMessage());

        } finally {
            // Temporarily disabled during testing.
            // long duration = System.currentTimeMillis() - startTime;
            // trafficLogger.logTraffic(tenantId, userId, path, duration, statusCode);
        }
    }

    private Optional<ResponseEntity<String>> applyPreProxyFilters(
            HttpServletRequest request,
            RouteDefinition matchedRoute) {

        Map<String, Map<String, String>> routeFilters = matchedRoute.filters();
        if (routeFilters == null) {
            return Optional.empty();
        }

        for (GatewayFilter filter : availableFilters) {
            System.out.println("PIPELINE TRIGGERED: Attempting to run filter -> " + filter.getName());

            if (!routeFilters.containsKey(filter.getName())) {
                continue;
            }

            Map<String, String> filterConfig = routeFilters.get(filter.getName());
            Optional<ResponseEntity<String>> filterResult = filter.filter(request, filterConfig);
            if (filterResult.isPresent()) {
                return filterResult;
            }
        }

        return Optional.empty();
    }
}
