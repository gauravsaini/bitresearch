// Ring All-Reduce over WebRTC — distributed gradient averaging
//
// The classic ring all-reduce algorithm works in 2 phases:
//   Phase 1 (Reduce-Scatter): Each node accumulates partial sums, sending chunks clockwise.
//                              After N-1 steps, each node holds the full sum for one chunk.
//   Phase 2 (All-Gather):     Each node shares its fully-reduced chunk, sending clockwise.
//                              After N-1 steps, every node has all fully-reduced chunks.
//
// Total communication: 2 * (N-1)/N * gradient_size
// This is bandwidth-optimal and doesn't require a central server.

import { WebRTCMesh, type RingTopology } from './webrtc-mesh';

export interface AllReduceConfig {
  numChunks?: number;     // split gradients into this many chunks (default: numPeers)
  timeoutMs?: number;     // timeout per phase (default: 10000)
  onProgress?: (phase: string, step: number, totalSteps: number) => void;
}

export class RingAllReduce {
  private mesh: WebRTCMesh;
  private config: AllReduceConfig;
  private currentStep: number = 0;

  // State for in-progress all-reduce
  private chunks: Float32Array[] = [];
  private receivedChunks = new Map<number, Float32Array>();
  private phaseResolvers = new Map<string, () => void>();
  private expectedReceives = 0;
  private receivedCount = 0;

  onLog: ((msg: string) => void) | null = null;

  constructor(mesh: WebRTCMesh, config: AllReduceConfig = {}) {
    this.mesh = mesh;
    this.config = {
      numChunks: config.numChunks,
      timeoutMs: config.timeoutMs || 10000,
      onProgress: config.onProgress,
    };

    // Register gradient receive handler
    this.mesh.onGradientsReceived = (fromPeerId, chunk, chunkIndex, phase) => {
      this.handleReceivedChunk(fromPeerId, chunk, chunkIndex, phase);
    };
  }

  private log(msg: string): void {
    console.log(`[AllReduce] ${msg}`);
    this.onLog?.(msg);
  }

  /**
   * Perform ring all-reduce on local gradients.
   * Returns the averaged gradients (same shape as input).
   *
   * @param localGradients - This worker's local gradients (flat Float32Array)
   * @param stepId - Training step ID for sequencing
   * @returns Averaged gradients across all peers
   */
  async allReduce(localGradients: Float32Array, stepId: number): Promise<Float32Array> {
    const ring = this.mesh.topology;
    if (!ring) {
      this.log('No ring topology — returning local gradients (no averaging)');
      return localGradients;
    }

    const N = ring.totalPeers;
    if (N <= 1) {
      return localGradients;
    }

    this.currentStep = stepId;
    const numChunks = this.config.numChunks || N;
    const chunkSize = Math.ceil(localGradients.length / numChunks);

    this.log(`Starting all-reduce: step=${stepId} | ${N} peers | ${numChunks} chunks | ${(localGradients.byteLength / 1024).toFixed(0)}KB total`);

    // Split gradients into chunks
    this.chunks = [];
    for (let i = 0; i < numChunks; i++) {
      const start = i * chunkSize;
      const end = Math.min(start + chunkSize, localGradients.length);
      this.chunks.push(new Float32Array(localGradients.buffer, start * 4, end - start));
    }

    // ── Phase 1: Reduce-Scatter ──────────────────────────────
    // N-1 steps. In each step k:
    //   - Send chunk[(myPos - k) mod N] to right neighbor
    //   - Receive a chunk from left neighbor, add to local chunk
    // After this phase, chunk[myPos] contains the sum across all peers.

    this.log('Phase 1: Reduce-Scatter');
    for (let step = 0; step < N - 1; step++) {
      const sendIdx = ((ring.self - step) % numChunks + numChunks) % numChunks;
      const recvIdx = ((ring.self - step - 1) % numChunks + numChunks) % numChunks;

      // Send our current chunk to right neighbor
      const success = this.mesh.sendGradientChunk(this.chunks[sendIdx], sendIdx, stepId, 'scatter');
      if (!success) {
        this.log(`Failed to send chunk ${sendIdx} in scatter step ${step}`);
      }

      // Wait for chunk from left neighbor
      const received = await this.waitForChunk(recvIdx, 'scatter', step);

      if (received) {
        // Accumulate: add received chunk to our local chunk
        const local = this.chunks[recvIdx];
        for (let i = 0; i < local.length; i++) {
          local[i] += received[i];
        }
      }

      this.config.onProgress?.('scatter', step + 1, N - 1);
    }

    // Now divide by N to get average
    const myChunkIdx = ring.self % numChunks;
    const myChunk = this.chunks[myChunkIdx];
    for (let i = 0; i < myChunk.length; i++) {
      myChunk[i] /= N;
    }

    // ── Phase 2: All-Gather ──────────────────────────────────
    // N-1 steps. In each step k:
    //   - Send the fully-reduced chunk to right neighbor
    //   - Receive a fully-reduced chunk from left neighbor
    // After this phase, all nodes have all fully-reduced chunks.

    this.log('Phase 2: All-Gather');
    for (let step = 0; step < N - 1; step++) {
      const sendIdx = ((ring.self - step + 1) % numChunks + numChunks) % numChunks;
      const recvIdx = ((ring.self - step) % numChunks + numChunks) % numChunks;

      // Send our fully-reduced chunk to right
      this.mesh.sendGradientChunk(this.chunks[sendIdx], sendIdx, stepId, 'gather');

      // Wait for chunk from left
      const received = await this.waitForChunk(recvIdx, 'gather', step);

      if (received) {
        // Replace our chunk with the received fully-reduced chunk
        this.chunks[recvIdx].set(received);
      }

      this.config.onProgress?.('gather', step + 1, N - 1);
    }

    // ── Reassemble ───────────────────────────────────────────
    const result = new Float32Array(localGradients.length);
    for (let i = 0; i < numChunks; i++) {
      const start = i * chunkSize;
      result.set(this.chunks[i], start);
    }

    this.log(`All-reduce complete for step ${stepId}`);
    return result;
  }

  // ── Receive Handling ──────────────────────────────────────────

  private handleReceivedChunk(
    _fromPeerId: string,
    chunk: Float32Array,
    chunkIndex: number,
    phase: 'scatter' | 'gather',
  ): void {
    const key = `${phase}-${chunkIndex}`;
    this.receivedChunks.set(chunkIndex, new Float32Array(chunk));

    // Resolve any waiting promise
    const resolver = this.phaseResolvers.get(key);
    if (resolver) {
      this.phaseResolvers.delete(key);
      resolver();
    }
  }

  private waitForChunk(
    chunkIndex: number,
    phase: 'scatter' | 'gather',
    _step: number,
  ): Promise<Float32Array | null> {
    const key = `${phase}-${chunkIndex}`;

    // Check if already received
    if (this.receivedChunks.has(chunkIndex)) {
      const data = this.receivedChunks.get(chunkIndex)!;
      this.receivedChunks.delete(chunkIndex);
      return Promise.resolve(data);
    }

    // Wait with timeout
    return new Promise<Float32Array | null>((resolve) => {
      const timeout = setTimeout(() => {
        this.phaseResolvers.delete(key);
        this.log(`Timeout waiting for chunk ${chunkIndex} in ${phase}`);
        resolve(null);
      }, this.config.timeoutMs);

      this.phaseResolvers.set(key, () => {
        clearTimeout(timeout);
        const data = this.receivedChunks.get(chunkIndex) || null;
        this.receivedChunks.delete(chunkIndex);
        resolve(data);
      });
    });
  }
}
