import Card from "./common/Card";

export default function ThreatLogTable({ logs }) {
  return (
    <Card title="Live Threat Log" className="fade-in-delay-3">
      <div className="log-table-wrap">
        <table>
          <thead>
            <tr>
              <th>Timestamp</th>
              <th>IP Address</th>
              <th>Endpoint</th>
              <th>Threat Type</th>
              <th>Score</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            {logs.length ? (
              logs.map((log) => (
                <tr key={log.id} className={log.action === "Block" ? "log-blocked" : ""}>
                  <td className="mono">{log.timestamp}</td>
                  <td className="mono">{log.ipAddress}</td>
                  <td>{log.endpoint}</td>
                  <td>{log.threatType}</td>
                  <td className="mono">{log.score.toFixed(1)}%</td>
                  <td>
                    <span className={`action-badge action-${log.action.toLowerCase()}`}>{log.action}</span>
                  </td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan={6} className="empty-row">
                  No threat activity yet. Start a simulation to stream telemetry.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </Card>
  );
}
