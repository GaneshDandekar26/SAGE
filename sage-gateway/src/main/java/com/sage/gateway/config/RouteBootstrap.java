package com.sage.gateway.config;

import com.sage.gateway.routing.RouteRegistry;
import com.sage.gateway.routing.RouteResolver;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.ApplicationArguments;
import org.springframework.boot.ApplicationRunner;
import org.springframework.stereotype.Component;

@Component
public class RouteBootstrap implements ApplicationRunner {

    private static final Logger logger = LoggerFactory.getLogger(RouteBootstrap.class);

    private final RouteRegistry routeRegistry;
    private final RouteResolver routeResolver;

    public RouteBootstrap(RouteRegistry routeRegistry, RouteResolver routeResolver) {
        this.routeRegistry = routeRegistry;
        this.routeResolver = routeResolver;
    }

    @Override
    public void run(ApplicationArguments args) {
        logger.info("Bootstrapping SAGE Gateway Radix Tree...");

        try {
            routeResolver.updateTree(routeRegistry.getRoot());
            logger.info("Radix tree compiled and deployed successfully.");
        } catch (Exception e) {
            logger.error("CRITICAL: Failed to build route tree on startup!", e);
        }
    }
}
