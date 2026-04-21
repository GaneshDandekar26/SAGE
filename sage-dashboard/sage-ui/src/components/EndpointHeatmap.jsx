import Card from "./common/Card";

export default function EndpointHeatmap({ hotspots }) {
  const maxHits = hotspots.length ? hotspots[0].hits : 1;

  return (
    <Card title="Endpoint Hotspots (Top 5)" className="fade-in-delay-3">
      <div className="endpoint-heatmap-list">
        {hotspots.length ? (
          hotspots.map((row) => (
            <div key={row.path} className="endpoint-heatmap-row">
              <div className="row-between">
                <span className="mono endpoint-path">{row.path}</span>
                <span className="mono endpoint-hits">{row.hits} hits</span>
              </div>
              <div className="endpoint-heatmap-track">
                <div
                  className={`endpoint-heatmap-fill threat-${row.dominantClass}`}
                  style={{ width: `${Math.max(8, (row.hits / maxHits) * 100)}%` }}
                />
              </div>
              <div className="endpoint-meta">
                <span className={`threat-chip threat-${row.dominantClass}`}>{row.dominantClass}</span>
              </div>
            </div>
          ))
        ) : (
          <p className="empty-row">No endpoint activity yet. Start a simulation to populate hotspots.</p>
        )}
      </div>
    </Card>
  );
}
