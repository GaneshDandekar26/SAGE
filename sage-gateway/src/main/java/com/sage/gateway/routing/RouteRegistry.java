package com.sage.gateway.routing;

import java.util.List;

public class RouteRegistry {
    private final RouteNode root = new RouteNode("/");

    // list of all raw definitions
    private final List<RouteDefinition> routes;

    public RouteRegistry(List<RouteDefinition> routes) {
        this.routes = routes;
        // Build the tree immediately upon startup
        for (RouteDefinition route : routes) {
            addRoute(route);
        }
    }

    private void addRoute(RouteDefinition routeDef) {
        String[] rawSegments = routeDef.pattern().split("/");
        RouteNode currentNode = root;

        for (String segment : rawSegments) {
            if (segment == null || segment.isEmpty()) continue;

            boolean isVariable = segment.startsWith("{") && segment.endsWith("}");

            if (isVariable) {
                String varName = segment.substring(1, segment.length() - 1);

                if (currentNode.variableChild == null) {
                    currentNode.variableChild = new RouteNode(segment);
                    currentNode.variableName = varName;
                }
                currentNode = currentNode.variableChild;
            } else {
                if (!currentNode.exactChildren.containsKey(segment)) {
                    currentNode.exactChildren.put(segment, new RouteNode(segment));
                }
                currentNode = currentNode.exactChildren.get(segment);
            }
        }
        currentNode.route = routeDef;
    }

    public RouteNode getRoot() {
        return root;
    }
}