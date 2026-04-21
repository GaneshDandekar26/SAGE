import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  scenarios: {
    scraper_sweep: {
      executor: 'constant-vus',
      vus: 60,
      duration: '35s',
    },
  },
};

const CRAWL_PATHS = [
  '/echo',
  '/api/benchmark',
  '/actuator/prometheus',
  '/api/get',
  '/api/test-route',
];

function randomIp() {
  return `172.${Math.floor(Math.random() * 31) + 16}.${Math.floor(Math.random() * 220) + 10}.${Math.floor(Math.random() * 220) + 10}`;
}

export default function () {
  const path = CRAWL_PATHS[Math.floor(Math.random() * CRAWL_PATHS.length)];
  const url = `http://localhost:8081${path}?p=${Math.floor(Math.random() * 1000)}`;

  const res = http.get(url, {
    headers: {
      'X-Forwarded-For': randomIp(),
      'User-Agent': 'ScraperBot/5.2',
    },
  });

  check(res, {
    'status captured': (r) => r.status > 0,
  });

  sleep(0.05);
}
