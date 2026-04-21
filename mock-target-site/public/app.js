async function loadProducts() {
  const container = document.getElementById("product-grid");

  try {
    const res = await fetch("/products");
    const products = await res.json();

    container.innerHTML = products
      .map(
        (p) => `
        <article class="card">
          <h3>${p.name}</h3>
          <div class="meta">
            <span>ID: ${p.id}</span>
            <span>Stock: ${p.inventory}</span>
          </div>
          <div class="price">$${p.price.toFixed(2)}</div>
        </article>
      `
      )
      .join("");
  } catch (err) {
    container.innerHTML = "<p>Could not load products.</p>";
  }
}

loadProducts();
