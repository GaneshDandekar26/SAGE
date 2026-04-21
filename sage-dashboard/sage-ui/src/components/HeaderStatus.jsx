function ServiceDot({ status }) {
  return <span className={`status-dot status-${status}`} aria-hidden="true" />;
}

export default function HeaderStatus({ services }) {
  return (
    <header className="dashboard-header fade-in">
      <div>
        <h1>SAGE Command Center</h1>
        <p>Threat telemetry, traffic controls, and ML defense insights</p>
      </div>
      <div className="service-grid">
        {services.map((service) => (
          <div key={service.name} className="service-item">
            <ServiceDot status={service.status} />
            <span>{service.name}</span>
            {service.meta ? <strong>{service.meta}</strong> : null}
          </div>
        ))}
      </div>
    </header>
  );
}
