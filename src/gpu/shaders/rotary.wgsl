// Rotary Position Embeddings (RoPE)
// Applies rotation to query/key tensors: [B, T, H, D]
// cos, sin: [1, T, 1, D/2]

struct RopeParams {
  B: u32,
  T: u32,
  H: u32,
  D: u32,
}

@group(0) @binding(0) var<storage, read_write> x: array<f32>;  // [B, T, H, D] — modified in place
@group(0) @binding(1) var<storage, read> cos_table: array<f32>; // [T, D/2]
@group(0) @binding(2) var<storage, read> sin_table: array<f32>; // [T, D/2]
@group(0) @binding(3) var<uniform> params: RopeParams;

@compute @workgroup_size(256)
fn apply_rotary(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  let half_D = params.D / 2u;
  let total_pairs = params.B * params.T * params.H * half_D;
  if (idx >= total_pairs) { return; }

  // Decode index
  let d = idx % half_D;
  let h = (idx / half_D) % params.H;
  let t = (idx / (half_D * params.H)) % params.T;
  let b = idx / (half_D * params.H * params.T);

  // Indices into x for the two halves
  let base = ((b * params.T + t) * params.H + h) * params.D;
  let i1 = base + d;
  let i2 = base + half_D + d;

  // Cos/sin for this position and dimension
  let rope_idx = t * half_D + d;
  let c = cos_table[rope_idx];
  let s = sin_table[rope_idx];

  let x1 = x[i1];
  let x2 = x[i2];

  // Apply rotation: [x1, x2] -> [x1*cos + x2*sin, x1*(-sin) + x2*cos]
  x[i1] = x1 * c + x2 * s;
  x[i2] = x1 * (-s) + x2 * c;
}

// Precompute rotary embedding tables
struct PrecomputeParams {
  seq_len: u32,
  head_dim: u32,
  base: f32,
  _pad: f32,
}

@group(0) @binding(0) var<storage, read_write> cos_out: array<f32>; // [seq_len, head_dim/2]
@group(0) @binding(1) var<storage, read_write> sin_out: array<f32>; // [seq_len, head_dim/2]
@group(0) @binding(2) var<uniform> precompute_params: PrecomputeParams;

@compute @workgroup_size(256)
fn precompute_rotary(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  let half_dim = precompute_params.head_dim / 2u;
  let total = precompute_params.seq_len * half_dim;
  if (idx >= total) { return; }

  let d = idx % half_dim;
  let t = idx / half_dim;

  let freq = 1.0 / pow(precompute_params.base, f32(2u * d) / f32(precompute_params.head_dim));
  let angle = f32(t) * freq;

  cos_out[idx] = cos(angle);
  sin_out[idx] = sin(angle);
}
