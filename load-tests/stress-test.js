import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
    stages: [
        { duration: '10s', target: 50 },   // Initial ramp-up phase to stabilize baseline load.
        { duration: '30s', target: 500 },  // Progressive scale-up to high concurrency.
        { duration: '1m', target: 500 },   // Sustained peak load for performance validation.
        { duration: '20s', target: 0 },    // Controlled ramp-down phase.
    ],
    thresholds: {
        'http_req_duration': ['p(99)<100'], // Target p99 latency under local execution constraints.
        'http_req_failed': ['rate<0.01'],
    },
};

export default function () {
    // Resolve target endpoint (typically local mock or echo service).
    const url = __ENV.URL || 'http://localhost:8081/echo';

    const res = http.get(url, {
        headers: { 'Connection': 'keep-alive' } // Enable connection reuse for consistent throughput behavior.
    });

    check(res, {
        'is 200': (r) => r.status === 200,
    });
}