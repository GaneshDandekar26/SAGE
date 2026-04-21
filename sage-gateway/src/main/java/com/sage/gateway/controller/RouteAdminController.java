package com.sage.gateway.controller;

import com.sage.gateway.routing.RouteDefinition;
import com.sage.gateway.service.RouteManagerService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/admin/routes")
public class RouteAdminController {
    private final RouteManagerService routeManagerService;

    public RouteAdminController(RouteManagerService routeManagerService) {
        this.routeManagerService = routeManagerService;
    }

    @PostMapping
    public ResponseEntity<String> addRoute(@RequestBody RouteDefinition newRoute) {
        try {
            routeManagerService.addRoute(newRoute);
            return ResponseEntity.ok("Route successfully added and dynamically loaded into SAGE.");
        } catch (Exception e) {
            return ResponseEntity.internalServerError().body("Failed to load route: " + e.getMessage());
        }
    }
}
