import { useEffect, useRef, useState } from "react";
import { io } from "socket.io-client";
import { clamp } from "../utils/formatters";

const BRIDGE_URL = import.meta.env.VITE_BRIDGE_URL || "http://localhost:6006";
const INITIAL_SERIES = Array.from({ length: 20 }, (_, idx) => ({ tick: idx + 1, rps: 0 }));

function getActionFromStatus(statusCode) {
  if (statusCode === 403) return "block";
  if (statusCode === 429) return "throttle";
  return "allow";
}

function mapThreatType(event) {
  const status = event?.response?.status;
  const probability = Number(event?.mlMetadata?.botProbability || 0);
  const mlThreatClass = event?.mlMetadata?.threatClass;
  const path = event?.request?.path || "";

  if (mlThreatClass && mlThreatClass !== "Benign") {
    if (mlThreatClass === "Flood") return "Flood Detection";
    if (mlThreatClass === "Infiltration") return "Infiltration Detection";
    if (mlThreatClass === "Bot") return "Bot Detection";
    return `${mlThreatClass} Detection`;
  }

  if (status === 429) return "Rate Limit Triggered";
  if (status === 403 && path.includes("prometheus")) return "Probe Blocking";
  if (status === 403 && probability >= 0.92) return "Flood Detection";
  if (status === 403) return "Bot Detection";
  if (probability >= 0.7) return "Anomalous Pattern";
  if (probability >= 0.45) return "Suspicious Session";
  return "Benign Session";
}

function toIsoTime(ms) {
  try {
    return new Date(ms).toLocaleTimeString();
  } catch {
    return new Date().toLocaleTimeString();
  }
}

function normalizeThreatClass(event, statusCode) {
  const mlClass = String(event?.mlMetadata?.threatClass || "").toLowerCase();
  const label = String(event?.label || "").toLowerCase();

  if (mlClass.includes("scraper") || mlClass === "bot") return "scraper";
  if (mlClass.includes("flood") || mlClass.includes("fastpathblock")) return "flood";
  if (mlClass.includes("infiltration") || mlClass.includes("recon") || mlClass.includes("probe")) {
    return "recon";
  }

  if (["human", "scraper", "flood", "recon"].includes(label)) {
    return label;
  }

  if (statusCode >= 400) {
    return "scraper";
  }

  return "human";
}

function dominantThreatClass(classCounts) {
  return Object.entries(classCounts).reduce((best, current) => {
    if (current[1] > best[1]) return current;
    return best;
  }, ["human", 0])[0];
}

export function useLiveTelemetry() {
  const [metrics, setMetrics] = useState({
    totalRequests: 0,
    threatsBlocked: 0,
    throttledRequests: 0,
    uptimeSeconds: 0,
  });
  const [actionTotals, setActionTotals] = useState({ allow: 0, throttle: 0, block: 0 });
  const [velocitySeries, setVelocitySeries] = useState(INITIAL_SERIES);
  const [features, setFeatures] = useState({
    sessionDepth: { value: 0, percent: 0, unit: "pkts" },
    temporalVariance: { value: 0, percent: 0, unit: "ms" },
    requestVelocity: { value: 0, percent: 0, unit: "req/s" },
    behavioralDiversity: { value: 0, percent: 0, unit: "paths" },
    endpointConcentration: { value: 0, percent: 0, unit: "ratio" },
    cartRatio: { value: 0, percent: 0, unit: "ratio" },
    assetSkipRatio: { value: 0, percent: 0, unit: "ratio" },
  });
  const [threatClassTotals, setThreatClassTotals] = useState({
    human: 0,
    scraper: 0,
    flood: 0,
    recon: 0,
  });
  const [endpointHotspots, setEndpointHotspots] = useState([]);
  const [mlConfidence, setMlConfidence] = useState(0);
  const [logs, setLogs] = useState([]);
  const [services, setServices] = useState([
    { name: "Kafka", status: "throttle" },
    { name: "Redis", status: "throttle" },
    { name: "Gateway", status: "throttle" },
    { name: "ML Service", status: "throttle", meta: "-- ms" },
  ]);
  const confidenceSamplesRef = useRef([]);

  const tickRef = useRef(20);
  const eventTimesRef = useRef([]);
  const endpointSetRef = useRef(new Set());
  const endpointStatsRef = useRef(new Map());
  const ipSessionDepthRef = useRef(new Map());
  const gapSamplesRef = useRef([]);
  const lastEventTimeRef = useRef(null);

  useEffect(() => {
    const socket = io(BRIDGE_URL, {
      transports: ["websocket"],
      reconnectionAttempts: 12,
      reconnectionDelay: 800,
    });

    socket.on("telemetry_update", (event) => {
      const statusCode = event?.response?.status || 200;
      const action = getActionFromStatus(statusCode);
      const botProbability = Number(event?.mlMetadata?.botProbability || 0);
      const timestamp = Number(event?.timestamp || Date.now());
      const ipAddress = event?.request?.ip || event?.userId || "unknown";
      const endpoint = event?.request?.path || "/unknown";
      const score = clamp(botProbability * 100, 0, 100);
      const threatClass = normalizeThreatClass(event, statusCode);

      setMetrics((prev) => ({
        totalRequests: prev.totalRequests + 1,
        threatsBlocked: prev.threatsBlocked + (action === "block" ? 1 : 0),
        throttledRequests: prev.throttledRequests + (action === "throttle" ? 1 : 0),
        uptimeSeconds: prev.uptimeSeconds,
      }));

      setActionTotals((prev) => ({
        allow: prev.allow + (action === "allow" ? 1 : 0),
        throttle: prev.throttle + (action === "throttle" ? 1 : 0),
        block: prev.block + (action === "block" ? 1 : 0),
      }));

      eventTimesRef.current.push(Date.now());
      eventTimesRef.current = eventTimesRef.current.filter((t) => Date.now() - t <= 5000);
      const rps = Number((eventTimesRef.current.length / 5).toFixed(1));

      tickRef.current += 1;
      setVelocitySeries((prev) => [...prev.slice(1), { tick: tickRef.current, rps }]);

      endpointSetRef.current.add(endpoint);
      const currentDepth = (ipSessionDepthRef.current.get(ipAddress) || 0) + 1;
      ipSessionDepthRef.current.set(ipAddress, currentDepth);

      if (lastEventTimeRef.current) {
        const gap = Math.abs(timestamp - lastEventTimeRef.current);
        gapSamplesRef.current.push(gap);
        gapSamplesRef.current = gapSamplesRef.current.slice(-50);
      }
      lastEventTimeRef.current = timestamp;

      const meanGap =
        gapSamplesRef.current.length > 0
          ? gapSamplesRef.current.reduce((acc, value) => acc + value, 0) / gapSamplesRef.current.length
          : 0;
      const sessionDepthValue = currentDepth * 12;
      const temporalVarianceValue = Number(meanGap.toFixed(1));
      const requestVelocityValue = rps;
      const behavioralDiversityValue = endpointSetRef.current.size;
      const rawFeatures = event?.features || {};
      const endpointConcentrationValue = Number(
        rawFeatures.endpointConcentration ?? rawFeatures.SAGE_Endpoint_Concentration ?? 0
      );
      const cartRatioValue = Number(rawFeatures.cartRatio ?? rawFeatures.SAGE_Cart_Ratio ?? 0);
      const assetSkipRatioValue = Number(rawFeatures.assetSkipRatio ?? rawFeatures.SAGE_Asset_Skip_Ratio ?? 0);

      setFeatures({
        sessionDepth: {
          value: Number(rawFeatures.sessionDepth ?? rawFeatures.SAGE_Session_Depth ?? sessionDepthValue),
          percent: clamp(
            (Number(rawFeatures.sessionDepth ?? rawFeatures.SAGE_Session_Depth ?? sessionDepthValue) / 12) * 100,
            0,
            100
          ),
          unit: "pkts",
        },
        temporalVariance: {
          value: Number(rawFeatures.temporalVariance ?? rawFeatures.SAGE_Temporal_Variance ?? temporalVarianceValue),
          percent: clamp(
            (Number(rawFeatures.temporalVariance ?? rawFeatures.SAGE_Temporal_Variance ?? temporalVarianceValue) / 1000) *
              100,
            0,
            100
          ),
          unit: "ms",
        },
        requestVelocity: {
          value: Number(rawFeatures.requestVelocity ?? rawFeatures.SAGE_Request_Velocity ?? requestVelocityValue),
          percent: clamp(
            (Number(rawFeatures.requestVelocity ?? rawFeatures.SAGE_Request_Velocity ?? requestVelocityValue) / 80) * 100,
            0,
            100
          ),
          unit: "req/s",
        },
        behavioralDiversity: {
          value: Number(
            rawFeatures.behavioralDiversity ?? rawFeatures.SAGE_Behavioral_Diversity ?? behavioralDiversityValue
          ),
          percent: clamp(
            (Number(
              rawFeatures.behavioralDiversity ?? rawFeatures.SAGE_Behavioral_Diversity ?? behavioralDiversityValue
            ) /
              15) *
              100,
            0,
            100
          ),
          unit: "paths",
        },
        endpointConcentration: {
          value: endpointConcentrationValue,
          percent: clamp(endpointConcentrationValue * 100, 0, 100),
          unit: "ratio",
        },
        cartRatio: {
          value: cartRatioValue,
          percent: clamp(cartRatioValue * 100, 0, 100),
          unit: "ratio",
        },
        assetSkipRatio: {
          value: assetSkipRatioValue,
          percent: clamp(assetSkipRatioValue * 100, 0, 100),
          unit: "ratio",
        },
      });

      setThreatClassTotals((prev) => ({
        ...prev,
        [threatClass]: prev[threatClass] + 1,
      }));

      const currentEndpoint = endpointStatsRef.current.get(endpoint) || {
        total: 0,
        classCounts: { human: 0, scraper: 0, flood: 0, recon: 0 },
      };
      currentEndpoint.total += 1;
      currentEndpoint.classCounts[threatClass] += 1;
      endpointStatsRef.current.set(endpoint, currentEndpoint);

      const topEndpoints = Array.from(endpointStatsRef.current.entries())
        .map(([path, stats]) => ({
          path,
          hits: stats.total,
          dominantClass: dominantThreatClass(stats.classCounts),
        }))
        .sort((a, b) => b.hits - a.hits)
        .slice(0, 5);
      setEndpointHotspots(topEndpoints);

      confidenceSamplesRef.current.push(score);
      confidenceSamplesRef.current = confidenceSamplesRef.current.slice(-12);
      const avgConfidence =
        confidenceSamplesRef.current.reduce((acc, value) => acc + value, 0) /
        confidenceSamplesRef.current.length;
      setMlConfidence(Number(avgConfidence.toFixed(1)));

      setLogs((prev) => {
        const next = [
          {
            id: event?.eventId || `${timestamp}-${Math.random().toString(36).slice(2, 8)}`,
            timestamp: toIsoTime(timestamp),
            ipAddress,
            endpoint,
            threatType: mapThreatType(event),
            score,
            action: action.charAt(0).toUpperCase() + action.slice(1),
          },
          ...prev,
        ];
        return next.slice(0, 50);
      });
    });

    socket.on("service_status", (status) => {
      setServices([
        { name: "Kafka", status: status?.kafka || "throttle" },
        { name: "Redis", status: status?.redis || "throttle" },
        { name: "Gateway", status: status?.gateway || "throttle" },
        {
          name: "ML Service",
          status: status?.ml || "throttle",
          meta: `${Math.round(status?.mlLatencyMs || 0)} ms`,
        },
      ]);
    });

    return () => {
      socket.close();
    };
  }, []);

  useEffect(() => {
    const timer = window.setInterval(() => {
      setMetrics((prev) => ({ ...prev, uptimeSeconds: prev.uptimeSeconds + 1 }));
    }, 1000);

    return () => window.clearInterval(timer);
  }, []);

  return {
    metrics,
    actionTotals,
    velocitySeries,
    features,
    threatClassTotals,
    endpointHotspots,
    mlConfidence,
    logs,
    services,
  };
}
