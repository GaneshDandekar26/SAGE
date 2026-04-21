package com.sage.gateway.routing;

import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.Map;

// record to hold the final destination data
public record RouteDefinition(
        @JsonProperty("id") String id,
        @JsonProperty("pattern") String pattern,
        @JsonProperty("backendUrl") String backendUrl,
        @JsonProperty("filters") Map<String, Map<String, String>> filters
) {
}
