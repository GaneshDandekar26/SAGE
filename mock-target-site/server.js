const express = require("express");
const path = require("path");

const app = express();
const PORT = 3001;
const CURRENCY = "USD";

const ADJECTIVES = [
  "Ultra",
  "Classic",
  "Premium",
  "Smart",
  "Eco",
  "Modern",
  "Compact",
  "Pro",
  "Essential",
  "Deluxe"
];

const PRODUCT_TYPES = [
  "Sneakers",
  "Headphones",
  "Backpack",
  "Jacket",
  "Watch",
  "Lamp",
  "Keyboard",
  "Chair",
  "Bottle",
  "Speaker"
];

const DESCRIPTORS = [
  "Series",
  "Edition",
  "Bundle",
  "Kit",
  "Pack",
  "Collection",
  "Line",
  "Model",
  "Set",
  "Range"
];

const WAREHOUSES = ["north-hub", "south-hub", "east-hub", "west-hub"];

function createProducts() {
  const list = [];
  for (let id = 1; id <= 50; id += 1) {
    const adjective = ADJECTIVES[id % ADJECTIVES.length];
    const type = PRODUCT_TYPES[(id * 3) % PRODUCT_TYPES.length];
    const descriptor = DESCRIPTORS[(id * 7) % DESCRIPTORS.length];

    const price = Number((12.99 + ((id * 11) % 180) + ((id * 17) % 100) / 100).toFixed(2));
    const inventory = (id * 19) % 140;

    list.push({
      id,
      name: `${adjective} ${type} ${descriptor}`,
      price,
      inventory,
      warehouse: WAREHOUSES[id % WAREHOUSES.length]
    });
  }
  return list;
}

const PRODUCTS = createProducts();
const PRODUCT_BY_ID = new Map(PRODUCTS.map((p) => [String(p.id), p]));

app.use(express.json());

app.use((req, res, next) => {
  const start = process.hrtime.bigint();
  res.on("finish", () => {
    const elapsedMs = Number(process.hrtime.bigint() - start) / 1e6;
    console.log(`${req.method} ${req.path} ${elapsedMs.toFixed(2)}ms`);
  });
  next();
});

app.use((req, res, next) => {
  const jitterMs = Math.floor(Math.random() * 16) + 5;
  setTimeout(next, jitterMs);
});

app.use("/static", express.static(path.join(__dirname, "public")));

app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "public", "index.html"));
});

app.get("/products", (req, res) => {
  const publicProducts = PRODUCTS.map(({ id, name, price, inventory }) => ({
    id,
    name,
    price,
    inventory
  }));
  res.json(publicProducts);
});

app.get("/products/:id", (req, res) => {
  const product = PRODUCT_BY_ID.get(req.params.id);
  if (!product) {
    return res.status(404).json({ error: "Product not found" });
  }

  const { id, name, price, inventory } = product;
  return res.json({ id, name, price, inventory });
});

app.get("/api/price/:id", (req, res) => {
  const product = PRODUCT_BY_ID.get(req.params.id);
  if (!product) {
    return res.status(404).json({ error: "Product not found" });
  }

  return res.json({ id: product.id, price: product.price, currency: CURRENCY });
});

app.get("/api/inventory/:id", (req, res) => {
  const product = PRODUCT_BY_ID.get(req.params.id);
  if (!product) {
    return res.status(404).json({ error: "Product not found" });
  }

  return res.json({ id: product.id, stock: product.inventory, warehouse: product.warehouse });
});

app.get("/api/search", (req, res) => {
  const q = String(req.query.q || "").trim().toLowerCase();
  if (!q) {
    return res.json([]);
  }

  const filtered = PRODUCTS
    .filter((p) => p.name.toLowerCase().includes(q))
    .map(({ id, name, price, inventory }) => ({ id, name, price, inventory }));

  return res.json(filtered);
});

app.get("/cart", (req, res) => {
  res.status(200).end();
});

app.post("/checkout", (req, res) => {
  res.status(200).json({ status: "ok" });
});

app.use((req, res) => {
  res.status(404).json({ error: "Not found" });
});

app.listen(PORT, () => {
  console.log(`Mock target site listening on http://localhost:${PORT}`);
});
