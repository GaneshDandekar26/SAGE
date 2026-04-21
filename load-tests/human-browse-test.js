import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  scenarios: {
    human_browsing: {
      executor: 'ramping-vus',
      startVUs: 2,
      stages: [
        { duration: '10s', target: 18 },
        { duration: '25s', target: 28 },
        { duration: '10s', target: 5 },
      ],
    },
  },
};

const PATHS = [
  '/echo',
  '/api/benchmark',
  '/api/get',
  '/actuator/prometheus',
];

function randomIp() {
  return `192.168.${Math.floor(Math.random() * 30) + 10}.${Math.floor(Math.random() * 220) + 10}`;
}

export default function () {
  const path = PATHS[Math.floor(Math.random() * PATHS.length)];
  const url = `http://localhost:8081${path}?session=${Math.floor(Math.random() * 90000)}`;

  const res = http.get(url, {
    headers: {
      'X-Forwarded-For': randomIp(),
      'User-Agent': 'Mozilla/5.0 HumanBrowser',
    },
  });

  check(res, {
    'response received': (r) => r.status > 0,
  });

  sleep(Math.random() * 1.6 + 0.4);
}
