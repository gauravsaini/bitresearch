#!/usr/bin/env node
// ── Smoke Test for BitResearch Signaling Server ──────────────────
// Validates server boot, HTTP endpoints, and (optionally) browser-based
// P2P training via Playwright.
//
// Usage: node scripts/smoke-test.mjs
// Exit codes: 0 = all checks passed, 1 = failure

import { spawn } from 'node:child_process';
import { readFileSync } from 'node:fs';
import http from 'node:http';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const ROOT = resolve(__dirname, '..');

// ── Config ──────────────────────────────────────────────────────
const SIGNALING_PORT = parseInt(process.env.SIGNALING_PORT || '8788');
const SIGNALING_URL = `http://localhost:${SIGNALING_PORT}`;
const BOOT_WAIT_MS = 2000;
const HTTP_TIMEOUT_MS = 5000;

// ── Helpers ─────────────────────────────────────────────────────

let passed = 0;
let failed = 0;

function pass(name) {
  passed++;
  console.log(`  ✅ PASS: ${name}`);
}

function fail(name, reason) {
  failed++;
  console.error(`  ❌ FAIL: ${name}`);
  console.error(`     Reason: ${reason}`);
}

function header(title) {
  console.log(`\n${'─'.repeat(50)}`);
  console.log(`  ${title}`);
  console.log(`${'─'.repeat(50)}`);
}

/** Make an HTTP GET and return parsed JSON (or throw on non-2xx / timeout). */
function httpGet(path) {
  return new Promise((resolve, reject) => {
    const url = `${SIGNALING_URL}${path}`;
    const timer = setTimeout(() => reject(new Error(`Timeout after ${HTTP_TIMEOUT_MS}ms: ${url}`)), HTTP_TIMEOUT_MS);

    http.get(url, (res) => {
      let body = '';
      res.on('data', (chunk) => (body += chunk));
      res.on('end', () => {
        clearTimeout(timer);
        if (res.statusCode < 200 || res.statusCode >= 300) {
          reject(new Error(`HTTP ${res.statusCode}: ${body.slice(0, 200)}`));
          return;
        }
        try {
          resolve(JSON.parse(body));
        } catch (e) {
          reject(new Error(`Invalid JSON: ${body.slice(0, 200)}`));
        }
      });
    }).on('error', (err) => {
      clearTimeout(timer);
      reject(err);
    });
  });
}

/** Wait for ms */
function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

/** Check if playwright is available as a dependency */
function hasPlaywright() {
  try {
    const pkg = JSON.parse(readFileSync(resolve(ROOT, 'package.json'), 'utf-8'));
    const allDeps = { ...pkg.dependencies, ...pkg.devDependencies };
    return 'playwright' in allDeps || '@playwright/test' in allDeps;
  } catch {
    return false;
  }
}

// ── Main ────────────────────────────────────────────────────────

async function main() {
  console.log('╔══════════════════════════════════════════════╗');
  console.log('║   BitResearch Smoke Test                     ║');
  console.log('╚══════════════════════════════════════════════╝');

  let serverProcess = null;

  try {
    // ── Step 1: Start signaling server ──────────────────────────
    header('Step 1: Starting signaling server');

    serverProcess = spawn('npx', ['tsx', 'server/signaling.ts'], {
      cwd: ROOT,
      stdio: ['ignore', 'pipe', 'pipe'],
      env: { ...process.env, SIGNALING_PORT: String(SIGNALING_PORT) },
    });

    serverProcess.stdout.on('data', (data) => {
      for (const line of data.toString().split('\n').filter(Boolean)) {
        console.log(`  [server] ${line}`);
      }
    });

    serverProcess.stderr.on('data', (data) => {
      for (const line of data.toString().split('\n').filter(Boolean)) {
        console.error(`  [server:err] ${line}`);
      }
    });

    // Wait for boot — poll until server is responding or timeout
    console.log(`  Waiting for server to become ready (max ${BOOT_WAIT_MS * 3}ms)...`);
    const bootDeadline = Date.now() + BOOT_WAIT_MS * 3;
    let serverReady = false;
    while (Date.now() < bootDeadline && !serverReady) {
      if (serverProcess.exitCode !== null) break;
      try {
        await httpGet('/status');
        serverReady = true;
      } catch {
        await sleep(300);
      }
    }

    if (serverProcess.exitCode !== null) {
      fail('Server startup', `Process exited with code ${serverProcess.exitCode}`);
      throw new Error('Server failed to start');
    }
    pass('Server started (process alive)');

    // ── Step 2: HTTP health checks ──────────────────────────────
    header('Step 2: HTTP endpoint checks');

    // Check /status
    try {
      const status = await httpGet('/status');
      if (typeof status.totalPeers === 'number' && status.totalPeers === 0) {
        pass(`/status → totalPeers: ${status.totalPeers}`);
      } else {
        fail('/status', `Expected totalPeers: 0, got ${JSON.stringify(status)}`);
      }

      if (Array.isArray(status.rooms)) {
        pass(`/status → rooms: ${JSON.stringify(status.rooms)}`);
      } else {
        fail('/status', `Expected rooms array, got ${typeof status.rooms}`);
      }

      if (typeof status.ringVersion === 'number') {
        pass(`/status → ringVersion: ${status.ringVersion}`);
      } else {
        fail('/status', `Expected ringVersion number, got ${typeof status.ringVersion}`);
      }
    } catch (e) {
      fail('/status endpoint', e.message);
    }

    // Check /turn-credentials
    try {
      const creds = await httpGet('/turn-credentials?peerId=test');

      if (typeof creds.username === 'string' && creds.username.includes(':')) {
        pass(`/turn-credentials → username: ${creds.username}`);
      } else {
        fail('/turn-credentials', `Expected username with timestamp:peerId format, got ${creds.username}`);
      }

      if (typeof creds.credential === 'string' && creds.credential.length > 0) {
        pass(`/turn-credentials → credential (HMAC): ${creds.credential.slice(0, 12)}...`);
      } else {
        fail('/turn-credentials', `Expected non-empty HMAC credential, got ${creds.credential}`);
      }

      if (Array.isArray(creds.urls) && creds.urls.length > 0) {
        pass(`/turn-credentials → urls: ${creds.urls.join(', ')}`);
      } else {
        fail('/turn-credentials', `Expected urls array, got ${JSON.stringify(creds.urls)}`);
      }

      if (typeof creds.expiresAt === 'number' && creds.expiresAt > Math.floor(Date.now() / 1000)) {
        pass(`/turn-credentials → expiresAt: ${new Date(creds.expiresAt * 1000).toISOString()}`);
      } else {
        fail('/turn-credentials', `Expected future expiresAt, got ${creds.expiresAt}`);
      }
    } catch (e) {
      fail('/turn-credentials endpoint', e.message);
    }

    // ── Step 3: Browser tests (Playwright) ──────────────────────
    if (hasPlaywright()) {
      header('Step 3: Browser-based P2P training test');
      console.log('  Playwright detected — running browser tests...');
      await runBrowserTests();
    } else {
      header('Step 3: Browser tests (skipped)');
      console.log('  Playwright not available, skipping browser tests');
    }

    // ── Summary ─────────────────────────────────────────────────
    header('Results');
    console.log(`  Passed: ${passed}`);
    console.log(`  Failed: ${failed}`);
    console.log('');

    if (failed > 0) {
      console.log('  ❌ SMOKE TEST FAILED');
      process.exitCode = 1;
    } else {
      console.log('  ✅ ALL SMOKE TESTS PASSED');
      process.exitCode = 0;
    }
  } catch (err) {
    console.error(`\n  💥 Fatal error: ${err.message}`);
    process.exitCode = 1;
  } finally {
    // ── Cleanup ─────────────────────────────────────────────────
    header('Cleanup');
    if (serverProcess && serverProcess.exitCode === null) {
      console.log('  Killing signaling server...');
      serverProcess.kill('SIGTERM');
      // Give it a moment to die gracefully
      await sleep(500);
      if (serverProcess.exitCode === null) {
        serverProcess.kill('SIGKILL');
      }
      console.log('  Server killed.');
    } else {
      console.log('  Server already exited.');
    }
  }
}

// ── Browser Tests (Playwright) ──────────────────────────────────

async function runBrowserTests() {
  let playwright;
  try {
    // Dynamic import so we don't crash if playwright is missing at import time
    playwright = await import('playwright');
  } catch (e) {
    console.log('  Playwright not available, skipping browser tests');
    return;
  }

  const { chromium } = playwright;
  let browser = null;
  let page1 = null;
  let page2 = null;

  try {
    // Launch headless Chromium
    browser = await chromium.launch({ headless: true });
    const context1 = await browser.newContext();
    const context2 = await browser.newContext();
    page1 = await context1.newPage();
    page2 = await context2.newPage();

    pass('Launched 2 headless Chromium tabs');

    // Minimal HTML that connects to signaling, forms ring, runs 3 training steps
    const testHtml = buildTestPage(SIGNALING_PORT);

    await page1.setContent(testHtml, { waitUntil: 'domcontentloaded' });
    await page2.setContent(testHtml, { waitUntil: 'domcontentloaded' });

    pass('Loaded test page in both tabs');

    // Wait for both peers to register and ring to form (signaling debounce is 1.5s)
    await sleep(4000);

    // Check peer registration via /status
    const status = await httpGet('/status');
    if (status.totalPeers >= 2) {
      pass(`Ring formed: ${status.totalPeers} peers connected`);
    } else {
      fail('Ring formation', `Expected >= 2 peers, got ${status.totalPeers}`);
    }

    // Wait for training steps to complete (3 steps * ~2s each)
    await sleep(8000);

    // Collect loss values from both pages
    const losses1 = await page1.evaluate(() => window.__smokeLosses || []);
    const losses2 = await page2.evaluate(() => window.__smokeLosses || []);

    console.log(`  Tab 1 losses: ${losses1.map((l) => l.toFixed(4)).join(', ')}`);
    console.log(`  Tab 2 losses: ${losses2.map((l) => l.toFixed(4)).join(', ')}`);

    // Verify loss decreasing (step 3 < step 1) on at least one tab
    const checkLossDecrease = (losses, label) => {
      if (losses.length >= 3) {
        if (losses[2] < losses[0]) {
          pass(`${label}: loss decreased (step1=${losses[0].toFixed(4)} → step3=${losses[2].toFixed(4)})`);
          return true;
        } else {
          fail(`${label}: loss did not decrease`, `step1=${losses[0].toFixed(4)}, step3=${losses[2].toFixed(4)}`);
          return false;
        }
      } else {
        fail(`${label}: not enough steps completed`, `Got ${losses.length}/3`);
        return false;
      }
    };

    checkLossDecrease(losses1, 'Tab 1');
    checkLossDecrease(losses2, 'Tab 2');
  } catch (e) {
    fail('Browser test', e.message);
  } finally {
    if (page1) await page1.close().catch(() => {});
    if (page2) await page2.close().catch(() => {});
    if (browser) await browser.close().catch(() => {});
    console.log('  Browser closed.');
  }
}

/** Build a minimal HTML page that connects to signaling and runs training steps */
function buildTestPage(port) {
  return `<!DOCTYPE html>
<html><head><meta charset="UTF-8"></head>
<body>
<div id="status">Initializing...</div>
<div id="losses"></div>
<script type="module">
  // Minimal smoke test — connects to signaling, forms ring, runs 3 training steps
  // Uses a tiny synthetic model (no WebGPU required — CPU fallback)

  import * as tf from 'http://localhost:5173/node_modules/@tensorflow/tfjs/dist/tf.esm.js';

  window.__smokeLosses = [];
  const peerId = 'smoke-' + Date.now() + '-' + Math.random().toString(36).slice(2, 6);
  const ws = new WebSocket('ws://localhost:${port}');

  function setStatus(msg) {
    const el = document.getElementById('status');
    if (el) el.textContent = msg;
    console.log('[Smoke ' + peerId.slice(-4) + '] ' + msg);
  }

  ws.onopen = () => {
    setStatus('Connected to signaling, registering...');
    ws.send(JSON.stringify({ type: 'register', peerId, room: 'smoke-test' }));
  };

  let ringFormed = false;
  let allPeers = [];

  ws.onmessage = async (event) => {
    const msg = JSON.parse(event.data);

    if (msg.type === 'ring-topology') {
      ringFormed = true;
      allPeers = msg.allPeers || [];
      setStatus('Ring formed! Position ' + msg.position + '/' + msg.totalPeers);

      // Run 3 training steps with a tiny model
      try {
        await tf.setBackend('cpu');
        await tf.ready();

        // Tiny model: 1-layer, 2-head, 32-embed, vocab=64, seq=8
        const model = tf.sequential();
        model.add(tf.layers.embedding({ inputDim: 64, outputDim: 32, inputLength: 8 }));
        model.add(tf.layers.flatten());
        model.add(tf.layers.dense({ units: 64, activation: 'softmax' }));
        model.compile({ optimizer: tf.train.adam(0.01), loss: 'categoricalCrossentropy' });

        for (let step = 1; step <= 3; step++) {
          const input = tf.randomUniform([2, 8], 0, 64, 'int32');
          const target = tf.oneHot(tf.randomUniform([2, 8], 0, 64, 'int32'), 64);
          const h = await model.fit(input, target, { epochs: 1, verbose: 0 });
          const loss = h.history.loss[0];
          window.__smokeLosses.push(loss);
          setStatus('Step ' + step + '/3 — loss=' + loss.toFixed(4));
          input.dispose();
          target.dispose();
        }

        setStatus('Done! Losses: ' + window.__smokeLosses.map(l => l.toFixed(4)).join(', '));
      } catch (e) {
        setStatus('Training error: ' + e.message);
        console.error(e);
      }
    }
  };

  ws.onerror = (e) => setStatus('WebSocket error');
  ws.onclose = () => setStatus('Disconnected');
</script>
</body></html>`;
}

// ── Run ─────────────────────────────────────────────────────────
main();
