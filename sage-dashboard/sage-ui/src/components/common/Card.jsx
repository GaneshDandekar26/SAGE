export default function Card({ title, children, className = "" }) {
  return (
    <section className={`card ${className}`.trim()}>
      {title ? <h3 className="card-title">{title}</h3> : null}
      {children}
    </section>
  );
}
