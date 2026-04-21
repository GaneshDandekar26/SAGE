import {
  CartesianGrid,
  RadialBar,
  RadialBarChart,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import Card from "../common/Card";

export default function TrafficVelocityChart({ data }) {
  const latestRps = data.length ? data[data.length - 1].rps : 0;
  const maxRps = Math.max(120, ...data.map((point) => point.rps || 0));
  const gaugeValue = Math.max(0, Math.min(100, (latestRps / maxRps) * 100));

  return (
    <Card title="Traffic Velocity (RPS)" className="fade-in-delay-2">
      <div className="velocity-layout">
        <div className="gauge-wrap">
          <ResponsiveContainer width="100%" height="100%">
            <RadialBarChart
              cx="50%"
              cy="70%"
              innerRadius="62%"
              outerRadius="92%"
              barSize={14}
              data={[{ name: "rps", value: gaugeValue, fill: "#2f8f4e" }]}
              startAngle={180}
              endAngle={0}
            >
              <RadialBar dataKey="value" cornerRadius={10} background />
            </RadialBarChart>
          </ResponsiveContainer>
          <div className="gauge-readout mono">
            <strong>{latestRps.toFixed(1)}</strong>
            <span>RPS</span>
          </div>
        </div>

        <div className="velocity-sparkline">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data}>
              <CartesianGrid stroke="#222228" vertical={false} />
              <XAxis dataKey="tick" tick={false} axisLine={false} tickLine={false} />
              <YAxis width={32} tick={{ fill: "#8f909a", fontSize: 10 }} axisLine={false} tickLine={false} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#141417",
                  border: "1px solid #222228",
                  borderRadius: "8px",
                  color: "#e8e8ec",
                }}
              />
              <Line type="monotone" dataKey="rps" stroke="#8fa0b8" strokeWidth={2} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </Card>
  );
}
