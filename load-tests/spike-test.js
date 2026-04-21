import http from 'k6/http';
import { check } from 'k6';

export const options = {
    // Defines a spike load profile to evaluate system elasticity under sudden traffic surges.
    stages: [
        { duration: '5s', target: 100 },   // Short warmup to initialize the k6 engine.
        { duration: '10s', target: 800 },  // Rapid scale-up to high concurrency.
        { duration: '20s', target: 800 },  // Sustained peak load to assess stability.
        { duration: '10s', target: 0 },    // Gradual ramp-down phase.
    ],
    thresholds: {
        // Latency tolerance adjusted for high-concurrency spike conditions.
        http_req_duration: ['p(99)<300'],
        http_req_failed: ['rate<0.05'],    // Acceptable failure rate during stress conditions.
    },
};

export default function () {
    // Target echo endpoint to measure raw gateway responsiveness under spike load.
    const url = __ENV.URL || 'http://localhost:8081/echo';

    const res = http.get(url, {
        headers: { 'Connection': 'keep-alive' } // Reuse connections to minimize transport overhead.
    });

    check(res, {
        'status is 200': (r) => r.status === 200,
    });
}