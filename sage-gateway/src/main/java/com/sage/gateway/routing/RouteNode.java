package com.sage.gateway.routing;

import java.util.HashMap;
import java.util.Map;

public class RouteNode {
    public String segment;
    public Map<String, RouteNode> exactChildren = new HashMap<>();

    public RouteNode variableChild;
    public String variableName;

    public RouteDefinition route;

    public RouteNode(String segment) {
        this.segment = segment;
    }
}