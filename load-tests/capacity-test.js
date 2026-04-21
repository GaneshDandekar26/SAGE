import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
    vus: 200,
    duration: '2m',
    thresholds: {
        http_req_duration: ['p(99)<100'],
        http_req_failed: ['rate<0.05'],
    },
};

export default function () {
    // Proxy endpoint under load test
    const url = 'http://localhost:8081/api/benchmark';

    const params = {
        headers: {
            'Content-Type': 'application/json',
            'Connection': 'keep-alive', // Enable connection reuse to simulate persistent client behavior
        },
    };

    const payload = JSON.stringify({
        event: 'capacity_test_v2',
        data: 'throughput_validation'
    });

    const res = http.post(url, payload, params);

    check(res, {
        'status is 200': (r) => r.status === 200,
        'not tripped': (r) => r.status !== 503,
    });

    // Minimal sleep reduces excessive CPU contention during high VU execution
    sleep(0.001);
}