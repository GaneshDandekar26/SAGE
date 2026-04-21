package com.sage.gateway.event;

public record RequestEvent(
        String eventId,
        long timestamp,
        String tenantId,
        String userId,
        String sessionId,
    String label,
        RequestDetails request,
        ResponseDetails response,
    FeatureVector features,
        MLMetadata mlMetadata
) {
    public record RequestDetails(String method, String path, String category, String ip) {}
    public record ResponseDetails(int status, long latencyMs) {}
    public record FeatureVector(
        double sessionDepth,
        double temporalVariance,
        double requestVelocity,
        double behavioralDiversity,
        double endpointConcentration,
        double cartRatio,
        double assetSkipRatio,
        double sequentialTraversal
    ) {}
    public record MLMetadata(double botProbability, int isBotFlag, String threatClass) {}
}