import { Cell, Pie, PieChart, ResponsiveContainer } from "recharts";
import Card from "../common/Card";

function getConfidenceLabel(score) {
  if (score >= 75) return "High Threat";
  if (score >= 45) return "Suspicious";
  return "Likely Benign";
}

export default function MLConfidenceChart({ score }) {
  const clampedScore = Math.max(0, Math.min(100, score));
  const data = [
    { name: "Score", value: clampedScore },
    { name: "Remaining", value: 100 - clampedScore },
  ];

  return (
    <Card title="ML Confidence" className="fade-in-delay-2">
      <div className="ml-wrap">
        <div className="chart-wrap small">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie data={data} dataKey="value" innerRadius={45} outerRadius={70} startAngle={90} endAngle={-270}>
                <Cell fill="#9e3d3d" />
                <Cell fill="#222228" />
              </Pie>
            </PieChart>
          </ResponsiveContainer>
        </div>
        <div className="ml-info mono">
          <p className="ml-score">{clampedScore.toFixed(1)}%</p>
          <p className="ml-label">{getConfidenceLabel(clampedScore)}</p>
        </div>
      </div>
    </Card>
  );
}
