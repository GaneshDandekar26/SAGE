import { useEffect, useMemo, useState } from "react";
import { clamp } from "../utils/formatters";

const ENDPOINTS = ["/api/login", "/api/search", "/api/export", "/api/billing", "/api/profile"];
const BOT_THREATS = ["Credential Stuffing", "Scraper Bot", "Token Abuse", "Brute Force"];
const HUMAN_THREATS = ["Benign Session", "Normal Navigation", "Authenticated Request"];

const INITIAL_STATE = {
  intensity: 45,
  metrics: {
    totalRequests: 0,
    threatsBlocked: 0,
    throttledRequests: 0,
    uptimeSeconds: 0,
  },
  actionTotals: {
    allow: 0,
    throttle: 0,
    block: 0,
  },
  velocitySeries: Array.from({ length: 20 }, (_, idx) => ({ tick: idx + 1, rps: 0 })),
  features: {
    sessionDepth: 12,
    temporalVariance: 15,
    requestVelocity: 10,
    behavioralDiversity: 20,
  },
  mlConfidence: 18,
  logs: [],
  services: {
    kafka: "allow",
    redis: "allow",
    gateway: "allow",
    ml: "allow",
    mlLatencyMs: 12,
  },
};

function randomItem(items) {
  return items[Math.floor(Math.random() * items.length)];
}

function randomIp() {
  return `${Math.floor(Math.random() * 200) + 10}.${Math.floor(Math.random() * 220) + 10}.${Math.floor(
    Math.random() * 220
  ) + 10}.${Math.floor(Math.random() * 220) + 10}`;
}

function buildLogEntry({ action, score, threatType }) {
  return {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    timestamp: new Date().toLocaleTimeString(),
    ipAddress: randomIp(),
    endpoint: randomItem(ENDPOINTS),
    threatType,
    score,
    action,
  };
}

export function useSimulation() {
  const [state, setState] = useState(INITIAL_STATE);
  const [processes, setProcesses] = useState({
    traffic: false,
    bot: false,
    human: false,
  });

  useEffect(() => {
    const interval = window.setInterval(() => {
      setState((prev) => {
        const activeCount = Number(processes.traffic) + Number(processes.bot) + Number(processes.human);
        const isRunning = activeCount > 0;
        const loadMultiplier = isRunning ? activeCount : 0;

        const rpsBase = isRunning ? prev.intensity / 2.5 : 0;
        const volatility = isRunning ? Math.random() * 12 - 6 : 0;
        const botBoost = processes.bot ? prev.intensity * 0.28 : 0;
        const humanBoost = processes.human ? prev.intensity * 0.12 : 0;
        const rps = clamp(rpsBase + volatility + botBoost + humanBoost, 0, 180);

        let allowRate = 0.78;
        let throttleRate = 0.15;
        let blockRate = 0.07;

        if (processes.bot) {
          allowRate -= 0.23;
          throttleRate += 0.15;
          blockRate += 0.08;
        }

        if (processes.human) {
          allowRate += 0.13;
          throttleRate -= 0.08;
          blockRate -= 0.05;
        }

        if (processes.traffic && !processes.bot && !processes.human) {
          throttleRate += 0.03;
        }

        allowRate = clamp(allowRate, 0.05, 0.95);
        throttleRate = clamp(throttleRate, 0.02, 0.6);
        blockRate = clamp(blockRate, 0.01, 0.55);

        const totalRate = allowRate + throttleRate + blockRate;
        allowRate /= totalRate;
        throttleRate /= totalRate;
        blockRate /= totalRate;

        const sampledRequests = Math.max(1, Math.round(rps * 0.7));
        const allowCount = isRunning ? Math.round(sampledRequests * allowRate) : 0;
        const throttleCount = isRunning ? Math.round(sampledRequests * throttleRate) : 0;
        const blockCount = isRunning ? Math.max(0, sampledRequests - allowCount - throttleCount) : 0;

        const baselineScore = processes.bot ? 72 : processes.human ? 28 : 45;
        const confidence = clamp(baselineScore + (Math.random() * 18 - 9), 2, 99);

        const nextFeatures = {
          sessionDepth: clamp(
            prev.features.sessionDepth + (processes.bot ? 4 : 1) + Math.random() * 8 - 4,
            1,
            100
          ),
          temporalVariance: clamp(
            prev.features.temporalVariance + (processes.bot ? 3 : 0.8) + Math.random() * 10 - 5,
            1,
            100
          ),
          requestVelocity: clamp(prev.features.requestVelocity + (rps / 20) + Math.random() * 6 - 3, 1, 100),
          behavioralDiversity: clamp(
            prev.features.behavioralDiversity + (processes.human ? 4 : 0.7) + Math.random() * 8 - 4,
            1,
            100
          ),
        };

        const logsToCreate = Math.min(3, Math.max(1, Math.round((rps / 140) * 3)));
        const newLogs = [];
        for (let index = 0; index < logsToCreate; index += 1) {
          const draw = Math.random();
          const action = draw < blockRate ? "Block" : draw < blockRate + throttleRate ? "Throttle" : "Allow";
          const score = clamp(
            confidence + (action === "Block" ? 12 : action === "Throttle" ? 6 : -18) + Math.random() * 10 - 5,
            1,
            99
          );
          const threatType =
            action === "Allow"
              ? randomItem(HUMAN_THREATS)
              : randomItem(BOT_THREATS.concat(["Burst Traffic", "Anomalous Pattern"]));

          newLogs.push(buildLogEntry({ action, score, threatType }));
        }

        const mlLatencyMs = clamp(
          Math.round(8 + loadMultiplier * 3 + prev.intensity * 0.06 + (Math.random() * 8 - 4)),
          8,
          120
        );

        return {
          ...prev,
          metrics: {
            totalRequests: prev.metrics.totalRequests + sampledRequests,
            threatsBlocked: prev.metrics.threatsBlocked + blockCount,
            throttledRequests: prev.metrics.throttledRequests + throttleCount,
            uptimeSeconds: prev.metrics.uptimeSeconds + 1,
          },
          actionTotals: {
            allow: prev.actionTotals.allow + allowCount,
            throttle: prev.actionTotals.throttle + throttleCount,
            block: prev.actionTotals.block + blockCount,
          },
          velocitySeries: [
            ...prev.velocitySeries.slice(1),
            {
              tick: prev.velocitySeries[prev.velocitySeries.length - 1].tick + 1,
              rps: Number(rps.toFixed(1)),
            },
          ],
          features: nextFeatures,
          mlConfidence: confidence,
          services: {
            kafka: isRunning ? "allow" : "throttle",
            redis: isRunning ? "allow" : "throttle",
            gateway: isRunning ? "allow" : "throttle",
            ml: confidence > 70 ? "block" : confidence > 45 ? "throttle" : "allow",
            mlLatencyMs,
          },
          logs: [...newLogs, ...prev.logs].slice(0, 50),
        };
      });
    }, 1000);

    return () => window.clearInterval(interval);
  }, [processes.bot, processes.human, processes.traffic]);

  const activeProcesses = useMemo(
    () => Object.entries(processes).filter(([, isActive]) => isActive).map(([name]) => name),
    [processes]
  );

  function setIntensity(value) {
    setState((prev) => ({ ...prev, intensity: clamp(value, 10, 100) }));
  }

  function launchTraffic() {
    setProcesses((prev) => ({ ...prev, traffic: true }));
  }

  function startBot() {
    setProcesses((prev) => ({ ...prev, bot: true, traffic: true }));
  }

  function startHuman() {
    setProcesses((prev) => ({ ...prev, human: true, traffic: true }));
  }

  function killAll() {
    setProcesses({ traffic: false, bot: false, human: false });
    setState(INITIAL_STATE);
  }

  const services = [
    { name: "Kafka", status: state.services.kafka },
    { name: "Redis", status: state.services.redis },
    { name: "Gateway", status: state.services.gateway },
    { name: "ML Service", status: state.services.ml, meta: `${state.services.mlLatencyMs} ms` },
  ];

  return {
    ...state,
    services,
    activeProcesses,
    launchTraffic,
    startBot,
    startHuman,
    killAll,
    setIntensity,
  };
}
