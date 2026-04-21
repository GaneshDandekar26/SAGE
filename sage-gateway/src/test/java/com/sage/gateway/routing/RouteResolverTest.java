package com.sage.gateway.routing;

import jakarta.servlet.http.HttpServletRequest;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

class RouteResolverTest {

    private RouteResolver routeResolver;

    @BeforeEach
    void setUp() {
        routeResolver = new RouteResolver();

        // Manually build a miniature Radix Tree for testing
        RouteNode root = new RouteNode("");

        // Branch A: Static Route -> /api/users
        RouteNode apiNode = new RouteNode("api");
        RouteNode usersNode = new RouteNode("users");
        usersNode.route = new RouteDefinition("test-route-id","/api/users", "https://backend-static", null);
        apiNode.exactChildren.put("users", usersNode);
        root.exactChildren.put("api", apiNode);

        // Branch B: Dynamic Route -> /api/users/{id}
        RouteNode idNode = new RouteNode("{id}");
        idNode.variableName = "id";
        idNode.route = new RouteDefinition("test-route-id","/api/test/{var}", "https://backend-dynamic", null);
        usersNode.variableChild = idNode;

        //  Load the test tree into the resolver
        routeResolver.updateTree(root);
    }

    @Test
    void shouldResolveExactStaticMatch() {
        HttpServletRequest request = mock(HttpServletRequest.class);

        RouteDefinition result = routeResolver.resolve("/api/users", request);

        assertNotNull(result, "Resolver should find the exact match");
        assertEquals("https://backend-static", result.backendUrl());
    }

    @Test
    void shouldResolveVariableMatchAndExtractVars() {
        HttpServletRequest request = mock(HttpServletRequest.class);

        RouteDefinition result = routeResolver.resolve("/api/users/999", request);

        assertNotNull(result, "Resolver should match the variable path");
        assertEquals("https://backend-dynamic", result.backendUrl());

        // Verify that the resolver caught "999" and saved it to the SAGE_VARS attribute
        verify(request).setAttribute(eq("SAGE_VARS"), anyMap());
    }

    @Test
    void shouldReturnNullFor404NotFound() {
        HttpServletRequest request = mock(HttpServletRequest.class);

        RouteDefinition result = routeResolver.resolve("/api/unknown-path", request);

        assertNull(result, "Resolver should return null for paths that do not exist");
    }
}