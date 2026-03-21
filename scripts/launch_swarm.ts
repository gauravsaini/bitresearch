/*
 * scripts/launch_swarm.ts — Launch distributed strategy swarm.
 *
 * Starts the signaling server + Vite dev server + opens N browser tabs
 * that auto-join the swarm and compute strategy variants.
 *
 * Usage:
 *   pnpm run swarm:headless              # 3 peers, headless
 *   tsx scripts/launch_swarm.ts --peers 5  # 5 peers
 *   tsx scripts/launch_swarm.ts --no-ui    # headless mode
 */

import { spawn, execSync } from 'child_process';
import * as path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');

const args = process.argv.slice(2);
const getArg = (name: string, fallback: string) => {
  const idx = args.indexOf(`--${name}`);
  return idx !== -1 && args[idx + 1] ? args[idx + 1] : fallback;
};
const hasFlag = (name: string) => args.includes(`--${name}`);

const PEERS = parseInt(getArg('peers', '3'));
const TIMEOUT = parseInt(getArg('timeout', '300'));
const NO_UI = hasFlag('no-ui');

console.log(`
🐝 finresearch — Distributed Strategy Swarm
   Peers: ${PEERS}
   Timeout: ${TIMEOUT}s
   Headless: ${NO_UI}
`);

// Start signaling server
console.log('Starting signaling server...');
const signaling = spawn('tsx', ['server/signaling.ts'], {
  cwd: ROOT,
  stdio: 'inherit',
});

// Start Vite dev server
console.log('Starting Vite dev server...');
const vite = spawn('npx', ['vite'], {
  cwd: ROOT,
  stdio: 'pipe',
});

let viteReady = false;
vite.stdout?.on('data', (data) => {
  const msg = data.toString();
  if (msg.includes('Local:') && !viteReady) {
    viteReady = true;
    console.log('Vite ready. Opening tabs...');
    openTabs();
  }
});

function openTabs() {
  const delay = 800;
  for (let i = 0; i < PEERS; i++) {
    setTimeout(() => {
      const url = `http://localhost:5173/swarm.html?autoStart=1${NO_UI ? '&hideUI=1' : ''}&peers=${PEERS}`;
      console.log(`  Opening tab ${i + 1}/${PEERS}: ${url}`);
      try {
        // Cross-platform browser open
        const cmd = process.platform === 'darwin' ? 'open' :
                    process.platform === 'win32' ? 'start' : 'xdg-open';
        execSync(`${cmd} "${url}"`, { stdio: 'ignore' });
      } catch {
        console.log(`  Could not open browser. Visit: ${url}`);
      }
    }, i * delay);
  }
}

// Timeout
if (TIMEOUT > 0) {
  setTimeout(() => {
    console.log(`\n⏰ Timeout reached (${TIMEOUT}s). Shutting down...`);
    signaling.kill();
    vite.kill();
    process.exit(0);
  }, TIMEOUT * 1000);
}

process.on('SIGINT', () => {
  console.log('\n🛑 Shutting down...');
  signaling.kill();
  vite.kill();
  process.exit(0);
});

process.on('SIGTERM', () => {
  signaling.kill();
  vite.kill();
  process.exit(0);
});
