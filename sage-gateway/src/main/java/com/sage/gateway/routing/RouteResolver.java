package com.sage.gateway.routing;

import jakarta.servlet.http.HttpServletRequest;
import org.springframework.stereotype.Component;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;

@Component
public class RouteResolver {

    private final AtomicReference<RouteNode> rootReference = new AtomicReference<>(new RouteNode(""));

    public void updateTree(RouteNode newRoot) {
        rootReference.set(newRoot);
    }

    public RouteDefinition resolve(String requestUri, HttpServletRequest request) {
        RouteNode currentNode = rootReference.get();

        String[] rawSegments = requestUri.split("/");


        for (String segment : rawSegments) {
            if (segment == null || segment.isEmpty()) continue;

            if (currentNode.exactChildren.containsKey(segment)) {
                currentNode = currentNode.exactChildren.get(segment);
            }
            else if (currentNode.variableChild != null) {
                // Store the variable in the request
                Map<String, String> vars = (Map<String, String>) request.getAttribute("SAGE_VARS");
                if (vars == null) vars = new HashMap<>();
                vars.put(currentNode.variableName, segment);
                request.setAttribute("SAGE_VARS", vars);

                currentNode = currentNode.variableChild;
            }
            else {
                return null; // 404 Not Found
            }
        }
        return currentNode.route; // Returns the route if it's a leaf, or null
    }
}