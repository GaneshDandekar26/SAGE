import Card from "../common/Card";
import { formatPercent } from "../../utils/formatters";

const ACTION_COLORS = {
  allow: "#2f8f4e",
  throttle: "#b7852e",
  block: "#9e3d3d",
};

export default function ActionDistributionChart({ totals }) {
  const total = totals.allow + totals.throttle + totals.block;
  const data = [
    { name: "Allow", key: "allow", value: totals.allow },
    { name: "Throttle", key: "throttle", value: totals.throttle },
    { name: "Block", key: "block", value: totals.block },
  ];

  return (
    <Card title="Action Distribution" className="fade-in-delay-2">
      <div className="distribution-wrap">
        <div className="distribution-bar-stack" aria-label="Action distribution segmented bar">
          {data.map((entry) => {
            const segmentPercent = total ? (entry.value / total) * 100 : 0;
            return (
              <span
                key={entry.key}
                className="distribution-segment"
                style={{
                  width: `${segmentPercent}%`,
                  backgroundColor: ACTION_COLORS[entry.key],
                }}
              />
            );
          })}
        </div>

        <div className="legend-list mono">
          {data.map((entry) => {
            const value = total ? (entry.value / total) * 100 : 0;
            return (
              <div key={entry.key} className="legend-item">
                <span className="legend-color" style={{ backgroundColor: ACTION_COLORS[entry.key] }} />
                <span>{entry.name}</span>
                <span>{entry.value}</span>
                <strong>{formatPercent(value)}</strong>
              </div>
            );
          })}
        </div>
      </div>
    </Card>
  );
}
