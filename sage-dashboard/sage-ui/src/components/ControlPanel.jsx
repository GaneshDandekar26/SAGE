import Card from "./common/Card";

const SERVICE_LABELS = {
  allow: "Online",
  throttle: "Degraded",
  block: "Offline",
};

export default function ControlPanel({ services, logCount }) {
  return (
    <aside className="control-column fade-in-delay-1">
      <Card title="Monitoring Scope" className="full-height">
        <div className="monitoring-note">
          Read-only mode enabled. This dashboard only visualizes live telemetry from Gateway, Kafka, Redis, and the ML service.
        </div>

        <div className="process-list-wrap">
          <h4>Service Health</h4>
          <ul className="process-list mono">
            {services.map((service) => (
              <li key={service.name}>
                {service.name}: {SERVICE_LABELS[service.status] || "Unknown"}
              </li>
            ))}
          </ul>
        </div>

        <div className="process-list-wrap">
          <h4>Stream Snapshot</h4>
          <ul className="process-list mono">
            <li>Live Threat Rows: {logCount}</li>
            <li>Telemetry Source: Kafka topic gateway-telemetry</li>
            <li>Interaction Mode: Observe-only</li>
          </ul>
        </div>
      </Card>
    </aside>
  );
}
