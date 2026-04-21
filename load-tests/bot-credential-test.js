import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  scenarios: {
    credential_stuffing: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '8s', target: 40 },
        { duration: '20s', target: 90 },
        { duration: '12s', target: 0 },
      ],
    },
  },
};

const TARGETS = [
  'http://localhost:8081/echo',
  'http://localhost:8081/api/benchmark',
  'http://localhost:8081/actuator/prometheus',
];

function randomIp() {
  return `10.${Math.floor(Math.random() * 220) + 10}.${Math.floor(Math.random() * 220) + 10}.${Math.floor(Math.random() * 220) + 10}`;
}

export default function () {
  const target = TARGETS[Math.floor(Math.random() * TARGETS.length)];
  const username = `admin_${Math.floor(Math.random() * 200)}`;

  const payload = JSON.stringify({
    username,
    password: 'Password123!',
    action: 'login_attempt',
  });

  const res = http.post(target, payload, {
    headers: {
      'Content-Type': 'application/json',
      'X-Forwarded-For': randomIp(),
      'User-Agent': 'CredentialBot/1.0',
    },
  });

  check(res, {
    'status captured': (r) => r.status > 0,
  });

  sleep(0.08);
}
