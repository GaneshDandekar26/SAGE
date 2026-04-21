import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
    scenarios: {
        warmup: {
            executor: 'shared-iterations',
            vus: 5,
            iterations: 50,
            maxDuration: '10s',
        },
        capacity_test: {
            executor: 'ramping-vus',
            startVUs: 0,
            stages: [
                { duration: '15s', target: 50 }, // Gradual ramp-up to stabilize resource utilization.
                { duration: '30s', target: 50 }, // Sustained load phase for capacity validation.
            ],
            startTime: '10s', // Delays main test execution until warmup completes.
        },
    },
    thresholds: {
        'http_req_duration{scenario:capacity_test}': ['p(99)<50'],
    },
};

export default function () {
    const url = 'http://localhost:8081/echo';

    // Generate randomized client IPs to distribute traffic across rate-limit keys.
    const randomIp = `10.${Math.floor(Math.random() * 255)}.${Math.floor(Math.random() * 255)}.${Math.floor(Math.random() * 255)}`;

    const params = {
        headers: {
            'Content-Type': 'application/json',
            'X-Forwarded-For': randomIp,
        },
    };

    const payload = JSON.stringify({ event: 'benchmark_test' });
    const res = http.post(url, payload, params);

    // Validate both successful processing and expected rate-limit responses.
    check(res, {
        'status is 200': (r) => r.status === 200,
        'rate limited (429)': (r) => r.status === 429,
    });

    // Introduce pacing between iterations to control request burst behavior.
    sleep(0.1);
}