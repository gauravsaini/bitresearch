// Signaling Server — lightweight peer discovery and WebRTC negotiation relay
// This server does NOT carry training data — it only helps peers find each other
// and exchange WebRTC offers/answers/ICE candidates.
// Once WebRTC connections are established, all gradient data flows P2P.

import { WebSocketServer, WebSocket } from 'ws';
import http from 'http';
import crypto from 'crypto';

interface RegisteredPeer {
  id: string;
  ws: WebSocket;
  joinedAt: number;
  ringPosition: number;
  room: string;             // room/session ID for isolated swarms
  peerType: 'compute' | 'observer';  // compute peers participate in ring, observers just watch
}

const PORT = parseInt(process.env.SIGNALING_PORT || '8788');
const MIN_PEERS_FOR_RING = 2;

// Map of room -> peers in that room
const rooms = new Map<string, Map<string, RegisteredPeer>>();
let ringVersion = 0;

// ── TURN Server HMAC Credential Rotation ──────────────────────

const TURN_SECRET = process.env.TURN_SECRET || 'default-turn-secret-change-me';
const TURN_REALM = process.env.TURN_REALM || 'bitresearch.training';
const TURN_HOST = process.env.TURN_HOST || 'localhost';
const TURN_PORT_NUM = parseInt(process.env.TURN_PORT || '3478');
const CREDENTIAL_TTL_SECONDS = parseInt(process.env.TURN_CRED_TTL || '86400'); // 24h

interface TurnCredentials {
  username: string;
  credential: string;
  urls: string[];
  expiresAt: number;
}

function generateTurnCredentials(peerId: string): TurnCredentials {
  const expiresAt = Math.floor(Date.now() / 1000) + CREDENTIAL_TTL_SECONDS;
  const username = `${expiresAt}:${peerId}`;
  const hmac = crypto.createHmac('sha1', TURN_SECRET);
  hmac.update(username);
  const credential = hmac.digest('base64');

  return {
    username,
    credential,
    urls: [
      `turn:${TURN_HOST}:${TURN_PORT_NUM}?transport=udp`,
      `turn:${TURN_HOST}:${TURN_PORT_NUM}?transport=tcp`,
    ],
    expiresAt,
  };
}

function log(msg: string): void {
  const ts = new Date().toISOString().slice(11, 19);
  console.log(`[${ts}] ${msg}`);
}

function getRoom(roomId: string): Map<string, RegisteredPeer> {
  if (!rooms.has(roomId)) {
    rooms.set(roomId, new Map());
  }
  return rooms.get(roomId)!;
}

// ── Ring Topology ─────────────────────────────────────────────

const ringFormTimers = new Map<string, ReturnType<typeof setTimeout>>();

function formRing(roomId: string): void {
  const allPeers = getRoom(roomId);
  // Only compute peers participate in the ring — observers just watch
  const peerList = Array.from(allPeers.values())
    .filter(p => p.peerType === 'compute')
    .sort((a, b) => a.id.localeCompare(b.id));

  if (peerList.length < MIN_PEERS_FOR_RING) {
    log(`⏳ Room ${roomId}: ${peerList.length} compute peers (of ${allPeers.size} total), need ${MIN_PEERS_FOR_RING} for ring`);
    // Notify all peers that ring is not formed yet
    for (const peer of allPeers.values()) {
      sendToPeer(peer.ws, { type: 'ring-waiting', computePeers: peerList.length, totalPeers: allPeers.size });
    }
    return;
  }

  ringVersion++;
  const N = peerList.length;
  log(`\n🔗 Room ${roomId}: Forming ring v${ringVersion} with ${N} peers:`);

  for (let i = 0; i < N; i++) {
    const peer = peerList[i];
    const leftIdx = (i - 1 + N) % N;
    const rightIdx = (i + 1) % N;
    peer.ringPosition = i;

    log(`   [${i}] ${peer.id.slice(-8)} ← ${peerList[leftIdx].id.slice(-8)} → ${peerList[rightIdx].id.slice(-8)}`);

    // Send ring topology + TURN credentials
    const turnCreds = generateTurnCredentials(peer.id);

    sendToPeer(peer.ws, {
      type: 'ring-topology',
      position: i,
      totalPeers: N,
      leftPeer: peerList[leftIdx].id,
      rightPeer: peerList[rightIdx].id,
      ringVersion,
      allPeers: peerList.map(p => ({ id: p.id, position: p.ringPosition })),
      turnServers: turnCreds,
    });
  }

  log(`✅ Room ${roomId}: Ring v${ringVersion} formed\n`);
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
    const allPeers: any[] = [];
    let computeCount = 0;
    for (const [roomId, peers] of rooms) {
      for (const p of peers.values()) {
        allPeers.push({ id: p.id, room: roomId, position: p.ringPosition, peerType: p.peerType, joinedAt: p.joinedAt });
        if (p.peerType === 'compute') computeCount++;
      }
    }
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      rooms: Array.from(rooms.keys()),
      peers: allPeers,
      ringVersion,
      totalPeers: allPeers.length,
      computePeers: computeCount,
      observerPeers: allPeers.length - computeCount,
    }));
  } else if (req.url?.startsWith('/turn-credentials')) {
    // REST endpoint for managed TURN credential rotation
    const url = new URL(req.url, `http://localhost:${PORT}`);
    const peerId = url.searchParams.get('peerId') || 'anonymous';
    const creds = generateTurnCredentials(peerId);
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(creds));
  } else {
    res.writeHead(200, { 'Content-Type': 'text/plain' });
    let totalPeers = 0, computePeers = 0;
    for (const peers of rooms.values()) {
      for (const p of peers.values()) {
        totalPeers++;
        if (p.peerType === 'compute') computePeers++;
      }
    }
    res.end(`WebRTC Signaling Server | ${computePeers} compute + ${totalPeers - computePeers} observer | ${rooms.size} rooms | ring v${ringVersion}`);
  }
});

const wss = new WebSocketServer({ server });

wss.on('connection', (ws: WebSocket) => {
  let peerId: string | null = null;
  let roomId: string = 'default';

  ws.on('message', (data: Buffer) => {
    try {
      const msg = JSON.parse(data.toString());

      switch (msg.type) {
        case 'register': {
          peerId = msg.peerId;
          roomId = msg.room || 'default';
          const peerType: 'compute' | 'observer' = msg.peerType === 'observer' ? 'observer' : 'compute';
          const peers = getRoom(roomId);

          peers.set(peerId!, {
            id: peerId!,
            ws,
            joinedAt: Date.now(),
            ringPosition: -1,
            room: roomId,
            peerType,
          });

          const computeCount = Array.from(peers.values()).filter(p => p.peerType === 'compute').length;
          log(`✅ Peer registered: ${peerId!.slice(-8)} [${peerType}] in room ${roomId} (${computeCount} compute, ${peers.size} total)`);

          // Reform ring with new peer (debounced per room)
          clearTimeout(ringFormTimers.get(roomId));
          ringFormTimers.set(roomId, setTimeout(() => formRing(roomId), 1500));
          break;
        }

        case 'offer':
        case 'answer':
        case 'ice-candidate': {
          // Relay to target peer (search across all rooms)
          for (const peers of rooms.values()) {
            const target = peers.get(msg.targetPeerId);
            if (target) {
              sendToPeer(target.ws, msg);
              break;
            }
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
    if (peerId) {
      const peers = getRoom(roomId);
      if (peers.has(peerId)) {
        const leavingPeer = peers.get(peerId)!;
        const wasCompute = leavingPeer.peerType === 'compute';
        peers.delete(peerId);
        log(`👋 Peer left: ${peerId.slice(-8)} [${leavingPeer.peerType}] from room ${roomId} (${peers.size} remaining)${wasCompute ? ' — ring will reform' : ''}`);

        // Notify others in the same room
        for (const peer of peers.values()) {
          sendToPeer(peer.ws, {
            type: 'peer-left',
            peerId,
          });
        }

        // Reform ring without this peer
        if (peers.size >= MIN_PEERS_FOR_RING) {
          clearTimeout(ringFormTimers.get(roomId));
          ringFormTimers.set(roomId, setTimeout(() => formRing(roomId), 1000));
        }

        // Clean up empty rooms
        if (peers.size === 0) {
          rooms.delete(roomId);
          clearTimeout(ringFormTimers.get(roomId));
          ringFormTimers.delete(roomId);
        }
      }
    }
  });
});

server.listen(PORT, () => {
  console.log(`
╔═══════════════════════════════════════════════════════════╗
║        🌐 WebRTC Signaling Server                        ║
╠═══════════════════════════════════════════════════════════╣
║                                                           ║
║  WebSocket: ws://localhost:${PORT}                          ║
║  Status:    http://localhost:${PORT}/status                  ║
║  TURN Creds: http://localhost:${PORT}/turn-credentials       ║
║                                                           ║
║  This server only relays WebRTC signaling messages.       ║
║  All training data flows peer-to-peer via WebRTC.         ║
║  Supports room-based isolated swarms.                     ║
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
