// WebRTC Peer-to-Peer Mesh — forms a ring topology for distributed all-reduce
// Each peer connects to its left and right neighbors in the ring.
// Gradients flow around the ring in O(N) steps for N workers.

export type PeerState = 'new' | 'connecting' | 'connected' | 'disconnected' | 'failed';

export interface PeerInfo {
  peerId: string;
  state: PeerState;
  connection: RTCPeerConnection | null;
  sendChannel: RTCDataChannel | null;
  recvChannel: RTCDataChannel | null;
  ringPosition: number;     // position in the ring (0..N-1)
  latencyMs: number;
  bytesReceived: number;
  bytesSent: number;
}

export interface RingTopology {
  self: number;            // my position in ring
  leftPeerId: string;     // peer I receive reduce-scatter from
  rightPeerId: string;    // peer I send reduce-scatter to
  totalPeers: number;
}

export type OnGradientsReceived = (fromPeerId: string, chunk: Float32Array, chunkIndex: number, phase: 'scatter' | 'gather') => void;
export type OnPeerStateChange = (peerId: string, state: PeerState) => void;
export type OnLog = (msg: string) => void;

const RTC_CONFIG: RTCConfiguration = {
  iceServers: [
    { urls: 'stun:stun.l.google.com:19302' },
    { urls: 'stun:stun1.l.google.com:19302' },
  ],
};

// Maximum message size for WebRTC data channels (64KB is safe across browsers)
const MAX_CHUNK_SIZE = 64 * 1024;

export class WebRTCMesh {
  private peers = new Map<string, PeerInfo>();
  private localPeerId: string;
  private signalingWs: WebSocket | null = null;
  private ring: RingTopology | null = null;

  onGradientsReceived: OnGradientsReceived | null = null;
  onPeerStateChange: OnPeerStateChange | null = null;
  onLog: OnLog | null = null;
  onRingFormed: ((ring: RingTopology) => void) | null = null;

  // Pending ICE candidates (received before remote description is set)
  private pendingCandidates = new Map<string, RTCIceCandidateInit[]>();

  constructor(peerId: string) {
    this.localPeerId = peerId;
  }

  get peerId(): string {
    return this.localPeerId;
  }

  get connectedPeerCount(): number {
    return Array.from(this.peers.values()).filter(p => p.state === 'connected').length;
  }

  get topology(): RingTopology | null {
    return this.ring;
  }

  private log(msg: string): void {
    console.log(`[WebRTC ${this.localPeerId.slice(-6)}] ${msg}`);
    this.onLog?.(msg);
  }

  // ── Signaling ───────────────────────────────────────────────────

  /** Connect to the signaling server for peer discovery */
  connectSignaling(signalingUrl: string): void {
    this.log(`Connecting to signaling server: ${signalingUrl}`);
    this.signalingWs = new WebSocket(signalingUrl);

    this.signalingWs.onopen = () => {
      this.log('Connected to signaling server');
      // Register ourselves
      this.sendSignaling({
        type: 'register',
        peerId: this.localPeerId,
        timestamp: Date.now(),
      });
    };

    this.signalingWs.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        this.handleSignalingMessage(msg);
      } catch (e) {
        this.log(`Signaling parse error: ${e}`);
      }
    };

    this.signalingWs.onclose = () => {
      this.log('Signaling server disconnected');
      // Reconnect after delay
      setTimeout(() => this.connectSignaling(signalingUrl), 3000);
    };

    this.signalingWs.onerror = (e) => {
      this.log(`Signaling error: ${e}`);
    };
  }

  private sendSignaling(msg: any): void {
    if (this.signalingWs?.readyState === WebSocket.OPEN) {
      this.signalingWs.send(JSON.stringify(msg));
    }
  }

  private async handleSignalingMessage(msg: any): Promise<void> {
    switch (msg.type) {
      case 'ring-topology': {
        // Server has assigned ring positions and told us our neighbors
        this.ring = {
          self: msg.position,
          leftPeerId: msg.leftPeer,
          rightPeerId: msg.rightPeer,
          totalPeers: msg.totalPeers,
        };
        this.log(`Ring formed! Position ${msg.position}/${msg.totalPeers} | left=${msg.leftPeer.slice(-6)} right=${msg.rightPeer.slice(-6)}`);
        this.onRingFormed?.(this.ring);

        // Initiate connections to ring neighbors
        // Higher peer ID initiates the connection (to avoid duplicates)
        if (this.localPeerId > msg.leftPeer) {
          await this.createPeerConnection(msg.leftPeer);
        }
        if (this.localPeerId > msg.rightPeer && msg.rightPeer !== msg.leftPeer) {
          await this.createPeerConnection(msg.rightPeer);
        }
        break;
      }

      case 'offer': {
        if (msg.targetPeerId !== this.localPeerId) return;
        this.log(`Received offer from ${msg.fromPeerId.slice(-6)}`);
        await this.handleOffer(msg.fromPeerId, msg.sdp);
        break;
      }

      case 'answer': {
        if (msg.targetPeerId !== this.localPeerId) return;
        this.log(`Received answer from ${msg.fromPeerId.slice(-6)}`);
        await this.handleAnswer(msg.fromPeerId, msg.sdp);
        break;
      }

      case 'ice-candidate': {
        if (msg.targetPeerId !== this.localPeerId) return;
        await this.handleIceCandidate(msg.fromPeerId, msg.candidate);
        break;
      }

      case 'peer-left': {
        this.log(`Peer ${msg.peerId.slice(-6)} left the ring`);
        this.removePeer(msg.peerId);
        break;
      }
    }
  }

  // ── WebRTC Connection Management ────────────────────────────────

  private async createPeerConnection(remotePeerId: string): Promise<void> {
    this.log(`Creating connection to ${remotePeerId.slice(-6)}`);

    const pc = new RTCPeerConnection(RTC_CONFIG);
    const peer: PeerInfo = {
      peerId: remotePeerId,
      state: 'connecting',
      connection: pc,
      sendChannel: null,
      recvChannel: null,
      ringPosition: -1,
      latencyMs: 0,
      bytesReceived: 0,
      bytesSent: 0,
    };
    this.peers.set(remotePeerId, peer);
    this.onPeerStateChange?.(remotePeerId, 'connecting');

    // Create data channel for sending gradients
    const sendChannel = pc.createDataChannel('gradients', {
      ordered: true,
      maxRetransmits: 3,
    });
    sendChannel.binaryType = 'arraybuffer';
    peer.sendChannel = sendChannel;

    sendChannel.onopen = () => {
      this.log(`Send channel to ${remotePeerId.slice(-6)} opened`);
      this.checkFullyConnected(remotePeerId);
    };

    sendChannel.onclose = () => {
      this.log(`Send channel to ${remotePeerId.slice(-6)} closed`);
    };

    // Handle incoming data channels
    pc.ondatachannel = (event) => {
      const recvChannel = event.channel;
      recvChannel.binaryType = 'arraybuffer';
      peer.recvChannel = recvChannel;

      recvChannel.onmessage = (e) => this.handleDataMessage(remotePeerId, e.data);
      recvChannel.onopen = () => {
        this.log(`Recv channel from ${remotePeerId.slice(-6)} opened`);
        this.checkFullyConnected(remotePeerId);
      };
    };

    // ICE candidate handling
    pc.onicecandidate = (event) => {
      if (event.candidate) {
        this.sendSignaling({
          type: 'ice-candidate',
          fromPeerId: this.localPeerId,
          targetPeerId: remotePeerId,
          candidate: event.candidate.toJSON(),
        });
      }
    };

    pc.onconnectionstatechange = () => {
      const state = pc.connectionState;
      this.log(`Connection to ${remotePeerId.slice(-6)}: ${state}`);
      if (state === 'failed' || state === 'disconnected') {
        peer.state = state === 'failed' ? 'failed' : 'disconnected';
        this.onPeerStateChange?.(remotePeerId, peer.state);
      }
    };

    // Create and send offer
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    this.sendSignaling({
      type: 'offer',
      fromPeerId: this.localPeerId,
      targetPeerId: remotePeerId,
      sdp: pc.localDescription?.toJSON(),
    });
  }

  private async handleOffer(fromPeerId: string, sdp: RTCSessionDescriptionInit): Promise<void> {
    let peer = this.peers.get(fromPeerId);

    if (!peer) {
      const pc = new RTCPeerConnection(RTC_CONFIG);
      peer = {
        peerId: fromPeerId,
        state: 'connecting',
        connection: pc,
        sendChannel: null,
        recvChannel: null,
        ringPosition: -1,
        latencyMs: 0,
        bytesReceived: 0,
        bytesSent: 0,
      };
      this.peers.set(fromPeerId, peer);

      // Create send channel
      const sendChannel = pc.createDataChannel('gradients', {
        ordered: true,
        maxRetransmits: 3,
      });
      sendChannel.binaryType = 'arraybuffer';
      peer.sendChannel = sendChannel;

      sendChannel.onopen = () => {
        this.log(`Send channel to ${fromPeerId.slice(-6)} opened`);
        this.checkFullyConnected(fromPeerId);
      };

      // Handle incoming data channels
      pc.ondatachannel = (event) => {
        const recvChannel = event.channel;
        recvChannel.binaryType = 'arraybuffer';
        peer!.recvChannel = recvChannel;

        recvChannel.onmessage = (e) => this.handleDataMessage(fromPeerId, e.data);
        recvChannel.onopen = () => {
          this.log(`Recv channel from ${fromPeerId.slice(-6)} opened`);
          this.checkFullyConnected(fromPeerId);
        };
      };

      pc.onicecandidate = (event) => {
        if (event.candidate) {
          this.sendSignaling({
            type: 'ice-candidate',
            fromPeerId: this.localPeerId,
            targetPeerId: fromPeerId,
            candidate: event.candidate.toJSON(),
          });
        }
      };

      pc.onconnectionstatechange = () => {
        const state = pc.connectionState;
        if (state === 'failed' || state === 'disconnected') {
          peer!.state = state === 'failed' ? 'failed' : 'disconnected';
          this.onPeerStateChange?.(fromPeerId, peer!.state);
        }
      };
    }

    const pc = peer.connection!;
    await pc.setRemoteDescription(new RTCSessionDescription(sdp));

    // Process any pending ICE candidates
    const pending = this.pendingCandidates.get(fromPeerId) || [];
    for (const candidate of pending) {
      await pc.addIceCandidate(new RTCIceCandidate(candidate));
    }
    this.pendingCandidates.delete(fromPeerId);

    const answer = await pc.createAnswer();
    await pc.setLocalDescription(answer);

    this.sendSignaling({
      type: 'answer',
      fromPeerId: this.localPeerId,
      targetPeerId: fromPeerId,
      sdp: pc.localDescription?.toJSON(),
    });
  }

  private async handleAnswer(fromPeerId: string, sdp: RTCSessionDescriptionInit): Promise<void> {
    const peer = this.peers.get(fromPeerId);
    if (!peer?.connection) return;

    await peer.connection.setRemoteDescription(new RTCSessionDescription(sdp));

    // Process pending ICE candidates
    const pending = this.pendingCandidates.get(fromPeerId) || [];
    for (const candidate of pending) {
      await peer.connection.addIceCandidate(new RTCIceCandidate(candidate));
    }
    this.pendingCandidates.delete(fromPeerId);
  }

  private async handleIceCandidate(fromPeerId: string, candidate: RTCIceCandidateInit): Promise<void> {
    const peer = this.peers.get(fromPeerId);
    if (peer?.connection?.remoteDescription) {
      await peer.connection.addIceCandidate(new RTCIceCandidate(candidate));
    } else {
      // Queue for later
      if (!this.pendingCandidates.has(fromPeerId)) {
        this.pendingCandidates.set(fromPeerId, []);
      }
      this.pendingCandidates.get(fromPeerId)!.push(candidate);
    }
  }

  private checkFullyConnected(peerId: string): void {
    const peer = this.peers.get(peerId);
    if (!peer) return;

    if (peer.sendChannel?.readyState === 'open') {
      // At least send channel is open — mark as connected
      peer.state = 'connected';
      this.log(`✅ Fully connected to ${peerId.slice(-6)}`);
      this.onPeerStateChange?.(peerId, 'connected');
    }
  }

  private removePeer(peerId: string): void {
    const peer = this.peers.get(peerId);
    if (peer) {
      peer.sendChannel?.close();
      peer.recvChannel?.close();
      peer.connection?.close();
      this.peers.delete(peerId);
      this.onPeerStateChange?.(peerId, 'disconnected');
    }
  }

  // ── Data Transfer ───────────────────────────────────────────────

  /** Handle incoming data from a peer */
  private handleDataMessage(fromPeerId: string, data: ArrayBuffer): void {
    const peer = this.peers.get(fromPeerId);
    if (peer) peer.bytesReceived += data.byteLength;

    // Parse message header
    const view = new DataView(data);
    const msgType = view.getUint8(0);
    // 0 = gradient chunk, 1 = ping, 2 = pong, 3 = sync barrier

    if (msgType === 0) {
      // Gradient chunk: [type(1) | phase(1) | chunkIndex(u32) | stepId(u32) | data(f32[])]
      const phase = view.getUint8(1) === 0 ? 'scatter' : 'gather';
      const chunkIndex = view.getUint32(2, true);
      // const stepId = view.getUint32(6, true);
      const gradientData = new Float32Array(data, 12);

      this.onGradientsReceived?.(fromPeerId, gradientData, chunkIndex, phase as 'scatter' | 'gather');
    } else if (msgType === 1) {
      // Ping — respond with pong
      this.sendToPeer(fromPeerId, new Uint8Array([2]).buffer);
    } else if (msgType === 2) {
      // Pong
      if (peer) {
        // Could track latency here
      }
    }
  }

  /** Send raw data to a peer */
  sendToPeer(peerId: string, data: ArrayBuffer): boolean {
    const peer = this.peers.get(peerId);
    if (!peer || peer.sendChannel?.readyState !== 'open') {
      return false;
    }

    try {
      // If data exceeds max chunk size, split it
      if (data.byteLength <= MAX_CHUNK_SIZE) {
        peer.sendChannel.send(data);
      } else {
        // Split into chunks
        const uint8 = new Uint8Array(data);
        for (let offset = 0; offset < uint8.byteLength; offset += MAX_CHUNK_SIZE) {
          const end = Math.min(offset + MAX_CHUNK_SIZE, uint8.byteLength);
          peer.sendChannel.send(uint8.slice(offset, end).buffer);
        }
      }
      peer.bytesSent += data.byteLength;
      return true;
    } catch (e) {
      this.log(`Send error to ${peerId.slice(-6)}: ${e}`);
      return false;
    }
  }

  /** Send a gradient chunk to the right neighbor in the ring */
  sendGradientChunk(
    gradients: Float32Array,
    chunkIndex: number,
    stepId: number,
    phase: 'scatter' | 'gather',
  ): boolean {
    if (!this.ring) return false;

    const targetPeerId = this.ring.rightPeerId;
    const headerSize = 12;
    const buf = new ArrayBuffer(headerSize + gradients.byteLength);
    const view = new DataView(buf);

    view.setUint8(0, 0); // type = gradient chunk
    view.setUint8(1, phase === 'scatter' ? 0 : 1);
    view.setUint32(2, chunkIndex, true);
    view.setUint32(6, stepId, true);

    new Float32Array(buf, headerSize).set(gradients);

    return this.sendToPeer(targetPeerId, buf);
  }

  /** Get info about all peers */
  getPeerInfos(): PeerInfo[] {
    return Array.from(this.peers.values());
  }

  /** Disconnect and clean up */
  disconnect(): void {
    for (const [id] of this.peers) {
      this.removePeer(id);
    }
    this.signalingWs?.close();
  }
}
