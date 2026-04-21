import Card from "./common/Card";

const FEATURE_LABELS = [
  ["sessionDepth", "Session Depth"],
  ["temporalVariance", "Temporal Variance"],
  ["requestVelocity", "Request Velocity"],
  ["behavioralDiversity", "Behavioral Diversity"],
  ["endpointConcentration", "Endpoint Concentration"],
  ["cartRatio", "Cart Ratio"],
  ["assetSkipRatio", "Asset Skip Ratio"],
];

export default function FeatureMonitor({ features }) {
  return (
    <Card title="Feature Monitor" className="fade-in-delay-3">
      <div className="feature-list">
        {FEATURE_LABELS.map(([key, label]) => {
          const feature = features[key] || { value: 0, percent: 0, unit: "" };
          return (
            <div key={key} className="feature-row">
              <div className="row-between">
                <span>{label}</span>
                <span className="mono">
                  {feature.value.toFixed(1)} {feature.unit}
                </span>
              </div>
              <div className="feature-bar-track">
                <div className="feature-bar-fill" style={{ width: `${feature.percent}%` }} />
              </div>
            </div>
          );
        })}
      </div>
    </Card>
  );
}
