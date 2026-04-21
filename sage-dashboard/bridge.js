const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const { Kafka, CompressionTypes, CompressionCodecs } = require('kafkajs');
const SnappyCodec = require('kafkajs-snappy');
const cors = require('cors');
const net = require('net');

CompressionCodecs[CompressionTypes.Snappy] = SnappyCodec;

const app = express();
app.use(cors());

const server = http.createServer(app);
const io = new Server(server, {
  cors: {
    origin: process.env.UI_ORIGIN || '*',
    methods: ['GET', 'POST'],
  },
});

const BRIDGE_PORT = Number(process.env.BRIDGE_PORT || 6006);
const KAFKA_BROKER = process.env.KAFKA_BROKER || 'localhost:9092';
const KAFKA_TOPIC = process.env.KAFKA_TOPIC || 'gateway-telemetry';
const KAFKA_GROUP_ID = process.env.KAFKA_GROUP_ID || `sage-dashboard-live-${process.pid}`;
const GATEWAY_HEALTH_URL = process.env.GATEWAY_HEALTH_URL || 'http://localhost:8081/echo';
const ML_HEALTH_URL = process.env.ML_HEALTH_URL || 'http://localhost:8000/docs';
const REDIS_HOST = process.env.REDIS_HOST || 'localhost';
const REDIS_PORT = Number(process.env.REDIS_PORT || 6379);

const kafka = new Kafka({
  clientId: 'sage-dashboard',
  brokers: [KAFKA_BROKER],
});

let consumer;
let kafkaConnected = false;
let reconnectDelayMs = 1000;
let reconnectTimer = null;
let isStartingConsumer = false;

function checkPort(host, port, timeoutMs = 900) {
  return new Promise((resolve) => {
    const socket = new net.Socket();
    let settled = false;
    const startedAt = Date.now();

    const finish = (ok) => {
      if (settled) return;
      settled = true;
      socket.destroy();
      resolve({ ok, latencyMs: Date.now() - startedAt });
    };

    socket.setTimeout(timeoutMs);
    socket.once('connect', () => finish(true));
    socket.once('timeout', () => finish(false));
    socket.once('error', () => finish(false));
    socket.connect(port, host);
  });
}

async function checkHttp(url) {
  const startedAt = Date.now();

  try {
    const response = await fetch(url, { method: 'GET' });
    return {
      ok: response.status < 500,
      latencyMs: Date.now() - startedAt,
    };
  } catch {
    return { ok: false, latencyMs: Date.now() - startedAt };
  }
}

function toStatus(ok, degraded = false) {
  if (!ok) return 'block';
  if (degraded) return 'throttle';
  return 'allow';
}

async function collectServiceStatus() {
  const [redis, gateway, ml] = await Promise.all([
    checkPort(REDIS_HOST, REDIS_PORT),
    checkHttp(GATEWAY_HEALTH_URL),
    checkHttp(ML_HEALTH_URL),
  ]);

  const payload = {
    kafka: toStatus(kafkaConnected),
    redis: toStatus(redis.ok),
    gateway: toStatus(gateway.ok, gateway.latencyMs > 350),
    ml: toStatus(ml.ok, ml.latencyMs > 500),
    mlLatencyMs: ml.latencyMs,
    timestamp: Date.now(),
  };

  io.emit('service_status', payload);
  return payload;
}

function createConsumer() {
  return kafka.consumer({ groupId: KAFKA_GROUP_ID });
}

function scheduleReconnect(reason) {
  if (reconnectTimer) {
    return;
  }

  kafkaConnected = false;
  console.error(`[Bridge] Kafka unavailable (${reason}). Retrying in ${reconnectDelayMs}ms`);

  reconnectTimer = setTimeout(async () => {
    reconnectTimer = null;
    reconnectDelayMs = Math.min(reconnectDelayMs * 2, 30000);
    await startKafkaConsumer();
  }, reconnectDelayMs);
}

async function startKafkaConsumer() {
  if (isStartingConsumer) {
    return;
  }

  isStartingConsumer = true;

  try {
    if (consumer) {
      try {
        await consumer.disconnect();
      } catch {
        // Ignore disconnect errors while replacing a stale consumer.
      }
    }

    consumer = createConsumer();

    consumer.on(consumer.events.CRASH, (e) => {
      const reason = e?.payload?.error?.message || 'consumer crash event';
      scheduleReconnect(reason);
    });

    await consumer.connect();
    await consumer.subscribe({ topic: KAFKA_TOPIC, fromBeginning: false });
    await consumer.run({
      eachMessage: async ({ message }) => {
        try {
          if (!message || !message.value) return;

          const telemetryEvent = JSON.parse(message.value.toString());
          console.log('[Kafka->Socket] Forwarding event:', {
            eventId: telemetryEvent.eventId,
            status: telemetryEvent.response?.status,
            path: telemetryEvent.request?.path,
          });

          io.emit('telemetry_update', telemetryEvent);
        } catch (error) {
          console.error('[Bridge] Failed to process Kafka message:', error);
        }
      },
    });

    kafkaConnected = true;
    reconnectDelayMs = 1000;
    console.log(`[+] Connected to Kafka Broker (${KAFKA_BROKER}) with group ${KAFKA_GROUP_ID}`);
  } catch (error) {
    const reason = error?.message || 'unknown Kafka error';
    scheduleReconnect(reason);
  } finally {
    isStartingConsumer = false;
  }
}

io.on('connection', (socket) => {
  console.log(`[+] Dashboard UI Connected: ${socket.id}`);
  collectServiceStatus().catch((error) => {
    console.error('[Health] Initial service status collection failed:', error);
  });

  socket.on('disconnect', () => {
    console.log(`[-] Dashboard UI Disconnected: ${socket.id}`);
  });
});

app.get('/api/status', async (_req, res) => {
  const services = await collectServiceStatus();
  res.json({ services });
});

setInterval(() => {
  collectServiceStatus().catch((error) => {
    console.error('[Health] Periodic service status collection failed:', error.message);
  });
}, 5000);

server.listen(BRIDGE_PORT, async () => {
  console.log(`[SAGE Bridge] WebSocket Server running on port ${BRIDGE_PORT}`);
  await startKafkaConsumer();
});

async function shutdown() {
  try {
    if (consumer) {
      await consumer.disconnect();
    }
  } catch {
    // Ignore disconnect errors during shutdown.
  }
  process.exit(0);
}

process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);
