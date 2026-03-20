// RMS Normalization: y = x / sqrt(mean(x^2) + eps)
// Operates on last dimension of [*, D] tensor

struct NormParams {
  size: u32,    // total elements
  D: u32,       // last dimension size
  eps: f32,
  _pad: f32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: NormParams;

@compute @workgroup_size(1)
fn rms_norm(@builtin(global_invocation_id) gid: vec3u) {
  let row = gid.x;
  let num_rows = params.size / params.D;
  if (row >= num_rows) { return; }

  let offset = row * params.D;

  // Compute mean of squares
  var sum_sq: f32 = 0.0;
  for (var i: u32 = 0u; i < params.D; i++) {
    let val = input[offset + i];
    sum_sq += val * val;
  }
  let rms = sqrt(sum_sq / f32(params.D) + params.eps);
  let inv_rms = 1.0 / rms;

  // Normalize
  for (var i: u32 = 0u; i < params.D; i++) {
    output[offset + i] = input[offset + i] * inv_rms;
  }
}

// RMS Norm backward: compute gradient w.r.t. input
// dx = (1/rms) * (dy - x * mean(x * dy) / (rms^2))
@group(0) @binding(3) var<storage, read> grad_output: array<f32>;
@group(0) @binding(4) var<storage, read_write> grad_input: array<f32>;

@compute @workgroup_size(1)
fn rms_norm_backward(@builtin(global_invocation_id) gid: vec3u) {
  let row = gid.x;
  let num_rows = params.size / params.D;
  if (row >= num_rows) { return; }

  let offset = row * params.D;

  // Recompute RMS
  var sum_sq: f32 = 0.0;
  for (var i: u32 = 0u; i < params.D; i++) {
    let val = input[offset + i];
    sum_sq += val * val;
  }
  let variance = sum_sq / f32(params.D);
  let rms = sqrt(variance + params.eps);
  let inv_rms = 1.0 / rms;

  // Compute x_hat = x / rms
  // Compute sum(dy * x_hat)
  var sum_dy_xhat: f32 = 0.0;
  for (var i: u32 = 0u; i < params.D; i++) {
    let x_hat = input[offset + i] * inv_rms;
    sum_dy_xhat += grad_output[offset + i] * x_hat;
  }

  // dx = inv_rms * (dy - x_hat * sum_dy_xhat / D)
  for (var i: u32 = 0u; i < params.D; i++) {
    let x_hat = input[offset + i] * inv_rms;
    grad_input[offset + i] = inv_rms * (grad_output[offset + i] - x_hat * sum_dy_xhat / f32(params.D));
  }
}
