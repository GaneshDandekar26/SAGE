import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
    vus: 50,             // Steady concurrent user load for endurance validation.
    duration: '10m',     // Extended execution window to observe memory stability.
    thresholds: {
        http_req_duration: ['p(99)<150'],
    },
};

export default function () {
    // Target the echo endpoint to evaluate raw JVM behavior without rate limiting influence.
    const url = 'http://localhost:8081/echo';

    const res = http.get(url, {
        headers: { 'Connection': 'keep-alive' } // Maintain persistent connections to reduce handshake overhead.
    });

    check(res, {
        'status is 200': (r) => r.status === 200,
    });

    // Controlled pacing between iterations to simulate realistic traffic patterns.
    sleep(0.05);
}