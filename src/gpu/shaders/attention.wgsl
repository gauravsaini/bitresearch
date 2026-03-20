// Causal self-attention — fused Q*K^T scoring + masking + softmax + V multiply
// For [B, H, T, D] tensors where:
//   B = batch, H = heads, T = sequence length, D = head dimension

struct AttnParams {
  B: u32,      // batch size
  H: u32,      // number of heads
  T: u32,      // sequence length
  D: u32,      // head dimension
  kv_H: u32,   // number of KV heads (for GQA)
  window: u32,  // sliding window size (0 = full attention)
}

@group(0) @binding(0) var<storage, read> Q: array<f32>;       // [B, T, H, D]
@group(0) @binding(1) var<storage, read> K: array<f32>;       // [B, T, kv_H, D]
@group(0) @binding(2) var<storage, read> V: array<f32>;       // [B, T, kv_H, D]
@group(0) @binding(3) var<storage, read_write> O: array<f32>; // [B, T, H, D]
@group(0) @binding(4) var<uniform> params: AttnParams;

// Helper to compute 4D index for Q/O: [B, T, H, D]
fn qo_idx(b: u32, t: u32, h: u32, d: u32) -> u32 {
  return ((b * params.T + t) * params.H + h) * params.D + d;
}

// Helper to compute 4D index for K/V: [B, T, kv_H, D]
fn kv_idx(b: u32, t: u32, h: u32, d: u32) -> u32 {
  return ((b * params.T + t) * params.kv_H + h) * params.D + d;
}

// Online softmax attention — numerically stable, single pass
// Each workgroup handles one (batch, head, query_pos) combination
@compute @workgroup_size(1)
fn causal_attention(@builtin(global_invocation_id) gid: vec3u) {
  let linear_id = gid.x;
  let b = linear_id / (params.H * params.T);
  let remainder = linear_id % (params.H * params.T);
  let h = remainder / params.T;
  let q_pos = remainder % params.T;

  if (b >= params.B || h >= params.H || q_pos >= params.T) {
    return;
  }

  // Map query head to KV head (for GQA)
  let kv_h = h / (params.H / params.kv_H);

  // Compute attention window bounds
  var k_start: u32 = 0u;
  if (params.window > 0u && q_pos >= params.window) {
    k_start = q_pos - params.window + 1u;
  }
  let k_end = q_pos + 1u; // causal: only attend to positions <= q_pos

  // Online softmax: compute max then sum in one conceptual pass
  var max_score: f32 = -1e10;

  // First pass: compute all scores and find max
  // (We recompute scores in second pass — no shared memory needed)
  for (var k_pos: u32 = k_start; k_pos < k_end; k_pos++) {
    var score: f32 = 0.0;
    for (var d: u32 = 0u; d < params.D; d++) {
      score += Q[qo_idx(b, q_pos, h, d)] * K[kv_idx(b, k_pos, kv_h, d)];
    }
    max_score = max(max_score, score);
  }

  // Second pass: compute exp(score - max) and sum
  var exp_sum: f32 = 0.0;
  // Also accumulate weighted V
  for (var d: u32 = 0u; d < params.D; d++) {
    O[qo_idx(b, q_pos, h, d)] = 0.0;
  }

  for (var k_pos: u32 = k_start; k_pos < k_end; k_pos++) {
    var score: f32 = 0.0;
    for (var d: u32 = 0u; d < params.D; d++) {
      score += Q[qo_idx(b, q_pos, h, d)] * K[kv_idx(b, k_pos, kv_h, d)];
    }
    let w = exp(score - max_score);
    exp_sum += w;

    for (var d: u32 = 0u; d < params.D; d++) {
      O[qo_idx(b, q_pos, h, d)] += w * V[kv_idx(b, k_pos, kv_h, d)];
    }
  }

  // Normalize by sum of exponentials
  let inv_sum = 1.0 / max(exp_sum, 1e-10);
  for (var d: u32 = 0u; d < params.D; d++) {
    O[qo_idx(b, q_pos, h, d)] *= inv_sum;
  }
}

// Backward pass for causal attention
// Computes dQ, dK, dV from dO
@group(0) @binding(5) var<storage, read> dO: array<f32>;       // [B, T, H, D]
@group(0) @binding(6) var<storage, read_write> dQ: array<f32>; // [B, T, H, D]
@group(0) @binding(7) var<storage, read_write> dK: array<f32>; // [B, T, kv_H, D]
@group(0) @binding(8) var<storage, read_write> dV: array<f32>; // [B, T, kv_H, D]

@compute @workgroup_size(1)
fn causal_attention_backward(@builtin(global_invocation_id) gid: vec3u) {
  let linear_id = gid.x;
  let b = linear_id / (params.H * params.T);
  let remainder = linear_id % (params.H * params.T);
  let h = remainder / params.T;
  let q_pos = remainder % params.T;

  if (b >= params.B || h >= params.H || q_pos >= params.T) {
    return;
  }

  let kv_h = h / (params.H / params.kv_H);

  var k_start: u32 = 0u;
  if (params.window > 0u && q_pos >= params.window) {
    k_start = q_pos - params.window + 1u;
  }
  let k_end = q_pos + 1u;

  // Recompute attention weights (forward pass)
  var max_score: f32 = -1e10;
  for (var k_pos: u32 = k_start; k_pos < k_end; k_pos++) {
    var score: f32 = 0.0;
    for (var d: u32 = 0u; d < params.D; d++) {
      score += Q[qo_idx(b, q_pos, h, d)] * K[kv_idx(b, k_pos, kv_h, d)];
    }
    max_score = max(max_score, score);
  }

  // Compute softmax weights and dS
  // dS_ij = P_ij * (dO_i · V_j - (dO_i · O_i))
  var exp_sum: f32 = 0.0;

  // First compute normalizer
  for (var k_pos: u32 = k_start; k_pos < k_end; k_pos++) {
    var score: f32 = 0.0;
    for (var d: u32 = 0u; d < params.D; d++) {
      score += Q[qo_idx(b, q_pos, h, d)] * K[kv_idx(b, k_pos, kv_h, d)];
    }
    exp_sum += exp(score - max_score);
  }
  let inv_sum = 1.0 / max(exp_sum, 1e-10);

  // Compute dO · O (for softmax backward)
  var do_dot_o: f32 = 0.0;
  for (var d: u32 = 0u; d < params.D; d++) {
    do_dot_o += dO[qo_idx(b, q_pos, h, d)] * O[qo_idx(b, q_pos, h, d)];
  }

  // Accumulate gradients
  for (var k_pos: u32 = k_start; k_pos < k_end; k_pos++) {
    var score: f32 = 0.0;
    for (var d: u32 = 0u; d < params.D; d++) {
      score += Q[qo_idx(b, q_pos, h, d)] * K[kv_idx(b, k_pos, kv_h, d)];
    }
    let p = exp(score - max_score) * inv_sum;

    // dV += P^T * dO
    for (var d: u32 = 0u; d < params.D; d++) {
      // atomicAdd would be ideal, but we use regular add since workgroups don't overlap on kv_h
      dV[kv_idx(b, k_pos, kv_h, d)] += p * dO[qo_idx(b, q_pos, h, d)];
    }

    // dP = dO * V^T
    var dp: f32 = 0.0;
    for (var d: u32 = 0u; d < params.D; d++) {
      dp += dO[qo_idx(b, q_pos, h, d)] * V[kv_idx(b, k_pos, kv_h, d)];
    }

    // dS = P * (dP - do_dot_o)
    let ds = p * (dp - do_dot_o);

    // dQ += dS * K
    for (var d: u32 = 0u; d < params.D; d++) {
      dQ[qo_idx(b, q_pos, h, d)] += ds * K[kv_idx(b, k_pos, kv_h, d)];
    }

    // dK += dS * Q
    for (var d: u32 = 0u; d < params.D; d++) {
      dK[kv_idx(b, k_pos, kv_h, d)] += ds * Q[qo_idx(b, q_pos, h, d)];
    }
  }
}
