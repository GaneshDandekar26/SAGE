import Card from "./common/Card";
import { formatNumber, formatUptime } from "../utils/formatters";

export default function GlobalMetrics({ metrics }) {
  return (
    <div className="metric-grid fade-in-delay-1">
      <Card title="Total Requests">
        <p className="metric-value">{formatNumber(metrics.totalRequests)}</p>
      </Card>
      <Card title="Threats Blocked">
        <p className="metric-value metric-danger">{formatNumber(metrics.threatsBlocked)}</p>
      </Card>
      <Card title="Throttled Requests">
        <p className="metric-value metric-warn">{formatNumber(metrics.throttledRequests)}</p>
      </Card>
      <Card title="Uptime">
        <p className="metric-value">{formatUptime(metrics.uptimeSeconds)}</p>
      </Card>
    </div>
  );
}
