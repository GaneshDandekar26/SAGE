package com.sage.gateway.service;

import com.sage.gateway.routing.RouteDefinition;
import com.sage.gateway.routing.RouteNode;
import com.sage.gateway.routing.RouteResolver;
import org.springframework.context.event.EventListener;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.context.event.ApplicationReadyEvent;
import org.springframework.context.event.EventListener;
@Service
public class RouteManagerService {
    private final RouteResolver routeResolver;
    private static final Logger logger = LoggerFactory.getLogger(RouteManagerService.class);
    private final List<RouteDefinition> activeRoutes = new ArrayList<>();

    public RouteManagerService(RouteResolver routeResolver) {
        this.routeResolver = routeResolver;
    }

    public synchronized void addRoute(RouteDefinition newRoute){
        activeRoutes.add(newRoute);
        rebuildAndDeployTree();
    }

    public void rebuildAndDeployTree(){
        RouteNode newRoot = new RouteNode("");

        for(RouteDefinition route : activeRoutes){
            insertIntoTree(newRoot, route);
        }

        routeResolver.updateTree(newRoot);
        System.out.println("SAGE Engine: Radix Tree Hot-Swapped! Total routes: " + activeRoutes.size());
    }

    private void insertIntoTree(RouteNode root, RouteDefinition route) {
        String[] rawSegments = route.pattern().split("/");
        RouteNode currentNode = root;

        for (String segment : rawSegments) {
            if (segment == null || segment.isEmpty()) continue;

            if (segment.startsWith("{") && segment.endsWith("}")) {

                String varName = segment.substring(1, segment.length() - 1);

                if (currentNode.variableChild == null) {
                    currentNode.variableChild = new RouteNode(segment);
                }

                currentNode.variableName = varName;

                currentNode = currentNode.variableChild;
            }
            // for exact static statements
            else {
                currentNode.exactChildren.putIfAbsent(segment, new RouteNode(segment));

                currentNode = currentNode.exactChildren.get(segment);
            }
        }
        currentNode.route = route;
    }


}
