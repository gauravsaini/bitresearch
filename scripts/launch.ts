#!/usr/bin/env tsx
// CLI Launcher — starts signaling server, vite, and opens browser tabs with auto-training
// Usage:
//   tsx scripts/launch.ts              # 3 peers, with UI
//   tsx scripts/launch.ts --peers 5    # 5 peers, with UI
//   tsx scripts/launch.ts --no-ui      # 3 peers, headless (minimal logging)
//   tsx scripts/launch.ts --timeout 300 # auto-stop after 300 seconds

import { spawn, ChildProcess } from 'child_process';
import { createInterface } from 'readline';

// ── CLI Arg Parsing ──────────────────────────────────────────
const args = process.argv.slice(2);
const peers = parseInt(args.find(a => a.startsWith('--peers'))?.split('=')[1]
  || args[args.indexOf('--peers') + 1]
  || '3');
const noUI = args.includes('--no-ui');
const timeout = parseInt(args.find(a => a.startsWith('--timeout'))?.split('=')[1]
  || args[args.indexOf('--timeout') + 1]
  || '0') || 0;
const maxSteps = parseInt(args.find(a => a.startsWith('--max-steps'))?.split('=')[1]
  || args[args.indexOf('--max-steps') + 1]
  || '0') || 0;

const VITE_PORT = 5173;
const SIGNALING_PORT = 8788;

const children: ChildProcess[] = [];

function log(msg: string): void {
  const ts = new Date().toISOString().slice(11, 19);
  console.log(`[${ts}] ${msg}`);
}

function cleanup(): void {
  log('Shutting down all processes...');
  for (const child of children) {
    try { child.kill('SIGTERM'); } catch {}
  }
  process.exit(0);
}

process.on('SIGINT', cleanup);
process.on('SIGTERM', cleanup);

// ── Wait for port to be ready ────────────────────────────────
async function waitForPort(port: number, name: string, timeoutMs = 15000): Promise<void> {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    try {
      const res = await fetch(`http://localhost:${port}/`);
      if (res.ok || res.status < 500) {
        log(`${name} ready on port ${port}`);
        return;
      }
    } catch {}
    await new Promise(r => setTimeout(r, 500));
  }
  throw new Error(`${name} did not start on port ${port} within ${timeoutMs}ms`);
}

// ── Open Browser Tabs ────────────────────────────────────────
function openBrowserTab(url: string): void {
  const platform = process.platform;
  let cmd: string;
  let cmdArgs: string[];

  if (platform === 'darwin') {
    cmd = 'open';
    cmdArgs = [url];
  } else if (platform === 'win32') {
    cmd = 'cmd';
    cmdArgs = ['/c', 'start', url];
  } else {
    cmd = 'xdg-open';
    cmdArgs = [url];
  }

  spawn(cmd, cmdArgs, { stdio: 'ignore', detached: true }).unref();
}

// ── Build URL params ─────────────────────────────────────────
function buildUrl(index: number): string {
  const params = new URLSearchParams();
  params.set('autoStart', '1');

  if (noUI) {
    params.set('hideUI', '1');
  }
  if (timeout > 0) {
    params.set('maxSeconds', timeout.toString());
  }
  if (maxSteps > 0) {
    params.set('maxSteps', maxSteps.toString());
  }

  const qs = params.toString();
  return `http://localhost:${VITE_PORT}/p2p.html?${qs}`;
}

// ── Main ─────────────────────────────────────────────────────
async function main(): Promise<void> {
  log(`BitResearch Trainer — launching ${peers} peer(s) ${noUI ? '(headless)' : '(with UI)'}`);
  if (timeout > 0) log(`Auto-stop after ${timeout}s`);
  if (maxSteps > 0) log(`Max steps: ${maxSteps}`);

  // 1. Start signaling server
  log('Starting signaling server...');
  const signaling = spawn('npx', ['tsx', 'server/signaling.ts'], {
    stdio: ['ignore', 'pipe', 'pipe'],
    env: { ...process.env, SIGNALING_PORT: String(SIGNALING_PORT) },
  });
  children.push(signaling);

  signaling.stdout?.on('data', (d) => {
    const msg = d.toString().trim();
    if (msg) console.log(`  [signaling] ${msg}`);
  });
  signaling.stderr?.on('data', (d) => {
    const msg = d.toString().trim();
    if (msg) console.error(`  [signaling] ${msg}`);
  });

  // 2. Start Vite dev server
  log('Starting Vite dev server...');
  const vite = spawn('npx', ['vite', '--port', String(VITE_PORT)], {
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  children.push(vite);

  vite.stdout?.on('data', (d) => {
    const msg = d.toString().trim();
    if (msg) console.log(`  [vite] ${msg}`);
  });
  vite.stderr?.on('data', (d) => {
    const msg = d.toString().trim();
    if (msg) console.error(`  [vite] ${msg}`);
  });

  // 3. Wait for both to be ready
  try {
    await Promise.all([
      waitForPort(SIGNALING_PORT, 'Signaling server'),
      waitForPort(VITE_PORT, 'Vite dev server'),
    ]);
  } catch (e) {
    console.error(`Failed to start: ${e}`);
    cleanup();
    return;
  }

  // 4. Open browser tabs
  log(`Opening ${peers} browser tab(s)...`);
  for (let i = 0; i < peers; i++) {
    const url = buildUrl(i);
    log(`  Tab ${i + 1}: ${url}`);
    openBrowserTab(url);
    // Stagger tab opens slightly to avoid race conditions
    if (i < peers - 1) {
      await new Promise(r => setTimeout(r, 800));
    }
  }

  log('');
  log('═══════════════════════════════════════════════════════');
  log('  Training swarm launched!');
  log(`  ${peers} peer(s) will auto-start training.`);
  log(`  Signaling: ws://localhost:${SIGNALING_PORT}`);
  log(`  Vite:      http://localhost:${VITE_PORT}`);
  if (!noUI) {
    log(`  Dashboard: http://localhost:${VITE_PORT}/index.html`);
  }
  if (timeout > 0) {
    log(`  Auto-stop: ${timeout}s`);
  }
  log('  Press Ctrl+C to stop.');
  log('═══════════════════════════════════════════════════════');
  log('');

  // Keep process alive
  await new Promise(() => {});
}

main().catch(e => {
  console.error(e);
  cleanup();
});
