// Signaling Server — lightweight peer discovery and WebRTC negotiation relay
// This server does NOT carry training data — it only helps peers find each other
// and exchange WebRTC offers/answers/ICE candidates.
// Once WebRTC connections are established, all gradient data flows P2P.

import { WebSocketServer, WebSocket } from 'ws';
import http from 'http';

interface RegisteredPeer {
  id: string;
  ws: WebSocket;
  joinedAt: number;
  ringPosition: number;
}

const PORT = parseInt(process.env.SIGNALING_PORT || '8788');
const MIN_PEERS_FOR_RING = 2;

const peers = new Map<string, RegisteredPeer>();
let ringVersion = 0;

function log(msg: string): void {
  const ts = new Date().toISOString().slice(11, 19);
  console.log(`[${ts}] ${msg}`);
}

// ── Ring Topology ─────────────────────────────────────────────

function formRing(): void {
  const peerList = Array.from(peers.values()).sort((a, b) => a.id.localeCompare(b.id));

  if (peerList.length < MIN_PEERS_FOR_RING) {
    log(`⏳ ${peerList.length} peers registered, need ${MIN_PEERS_FOR_RING} for ring`);
    return;
  }

  ringVersion++;
  const N = peerList.length;
  log(`\n🔗 Forming ring v${ringVersion} with ${N} peers:`);

  for (let i = 0; i < N; i++) {
    const peer = peerList[i];
    const leftIdx = (i - 1 + N) % N;
    const rightIdx = (i + 1) % N;
    peer.ringPosition = i;

    log(`   [${i}] ${peer.id.slice(-8)} ← ${peerList[leftIdx].id.slice(-8)} → ${peerList[rightIdx].id.slice(-8)}`);

    // Tell each peer about its neighbors
    sendToPeer(peer.ws, {
      type: 'ring-topology',
      position: i,
      totalPeers: N,
      leftPeer: peerList[leftIdx].id,
      rightPeer: peerList[rightIdx].id,
      ringVersion,
      allPeers: peerList.map(p => ({ id: p.id, position: p.ringPosition })),
    });
  }

  log(`✅ Ring v${ringVersion} formed\n`);
}

function sendToPeer(ws: WebSocket, msg: any): void {
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(msg));
  }
}

// ── HTTP + WebSocket Server ───────────────────────────────────

const server = http.createServer((req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');

  if (req.url === '/status') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      peers: Array.from(peers.values()).map(p => ({
        id: p.id,
        position: p.ringPosition,
        joinedAt: p.joinedAt,
      })),
      ringVersion,
      totalPeers: peers.size,
    }));
  } else {
    res.writeHead(200, { 'Content-Type': 'text/plain' });
    res.end(`WebRTC Signaling Server | ${peers.size} peers | ring v${ringVersion}`);
  }
});

const wss = new WebSocketServer({ server });

wss.on('connection', (ws: WebSocket) => {
  let peerId: string | null = null;

  ws.on('message', (data: Buffer) => {
    try {
      const msg = JSON.parse(data.toString());

      switch (msg.type) {
        case 'register': {
          peerId = msg.peerId;
          peers.set(peerId!, {
            id: peerId!,
            ws,
            joinedAt: Date.now(),
            ringPosition: -1,
          });
          log(`✅ Peer registered: ${peerId!.slice(-8)} (${peers.size} total)`);

          // Reform ring with new peer
          clearTimeout(ringFormTimer);
          ringFormTimer = setTimeout(() => formRing(), 1500);
          break;
        }

        case 'offer':
        case 'answer':
        case 'ice-candidate': {
          // Relay to target peer
          const target = peers.get(msg.targetPeerId);
          if (target) {
            sendToPeer(target.ws, msg);
          } else {
            log(`⚠️ Target peer ${msg.targetPeerId?.slice(-8)} not found for ${msg.type}`);
          }
          break;
        }

        default:
          log(`Unknown message type: ${msg.type}`);
      }
    } catch (e) {
      log(`Parse error: ${e}`);
    }
  });

  ws.on('close', () => {
    if (peerId && peers.has(peerId)) {
      peers.delete(peerId);
      log(`👋 Peer left: ${peerId.slice(-8)} (${peers.size} remaining)`);

      // Notify others
      for (const peer of peers.values()) {
        sendToPeer(peer.ws, {
          type: 'peer-left',
          peerId,
        });
      }

      // Reform ring without this peer
      if (peers.size >= MIN_PEERS_FOR_RING) {
        clearTimeout(ringFormTimer);
        ringFormTimer = setTimeout(() => formRing(), 1000);
      }
    }
  });
});

let ringFormTimer: ReturnType<typeof setTimeout>;

server.listen(PORT, () => {
  console.log(`
╔═══════════════════════════════════════════════════════════╗
║        🌐 WebRTC Signaling Server                        ║
╠═══════════════════════════════════════════════════════════╣
║                                                           ║
║  WebSocket: ws://localhost:${PORT}                          ║
║  Status:    http://localhost:${PORT}/status                  ║
║                                                           ║
║  This server only relays WebRTC signaling messages.       ║
║  All training data flows peer-to-peer via WebRTC.         ║
║                                                           ║
║  Waiting for peers to register...                         ║
╚═══════════════════════════════════════════════════════════╝
  `);
});

process.on('SIGINT', () => {
  console.log('\n🛑 Shutting down signaling server...');
  server.close();
  process.exit(0);
});
