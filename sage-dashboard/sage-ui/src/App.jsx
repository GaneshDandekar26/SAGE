import HeaderStatus from "./components/HeaderStatus";
import GlobalMetrics from "./components/GlobalMetrics";
import ControlPanel from "./components/ControlPanel";
import TrafficVelocityChart from "./components/charts/TrafficVelocityChart";
import ActionDistributionChart from "./components/charts/ActionDistributionChart";
import MLConfidenceChart from "./components/charts/MLConfidenceChart";
import ThreatClassBreakdownChart from "./components/charts/ThreatClassBreakdownChart";
import FeatureMonitor from "./components/FeatureMonitor";
import EndpointHeatmap from "./components/EndpointHeatmap";
import ThreatLogTable from "./components/ThreatLogTable";
import { useLiveTelemetry } from "./hooks/useLiveTelemetry";

export default function App() {
  const {
    metrics,
    actionTotals,
    velocitySeries,
    features,
    threatClassTotals,
    endpointHotspots,
    mlConfidence,
    logs,
    services,
  } = useLiveTelemetry();

  return (
    <div className="app-shell">
      <div className="dashboard-layout">
        <ControlPanel
          services={services}
          logCount={logs.length}
        />

        <main className="main-column">
          <HeaderStatus services={services} />
          <GlobalMetrics metrics={metrics} />

          <section className="chart-grid">
            <TrafficVelocityChart data={velocitySeries} />
            <ActionDistributionChart totals={actionTotals} />
            <MLConfidenceChart score={mlConfidence} />
          </section>

          <section className="insight-grid">
            <ThreatClassBreakdownChart totals={threatClassTotals} />
            <EndpointHeatmap hotspots={endpointHotspots} />
          </section>

          <section className="lower-grid">
            <FeatureMonitor features={features} />
            <ThreatLogTable logs={logs} />
          </section>
        </main>
      </div>
    </div>
  );
}