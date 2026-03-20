// Element-wise operations: add, mul, relu, squared relu, tanh, sigmoid, scale

struct Params {
  size: u32,
  scalar: f32,
}

@group(0) @binding(0) var<storage, read> input_a: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn add_scalar(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  if (idx >= params.size) { return; }
  output[idx] = input_a[idx] + params.scalar;
}

@compute @workgroup_size(256)
fn mul_scalar(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  if (idx >= params.size) { return; }
  output[idx] = input_a[idx] * params.scalar;
}

@compute @workgroup_size(256)
fn relu(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  if (idx >= params.size) { return; }
  output[idx] = max(input_a[idx], 0.0);
}

@compute @workgroup_size(256)
fn squared_relu(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  if (idx >= params.size) { return; }
  let r = max(input_a[idx], 0.0);
  output[idx] = r * r;
}

@compute @workgroup_size(256)
fn tanh_op(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  if (idx >= params.size) { return; }
  output[idx] = tanh(input_a[idx]);
}

@compute @workgroup_size(256)
fn sigmoid(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  if (idx >= params.size) { return; }
  output[idx] = 1.0 / (1.0 + exp(-input_a[idx]));
}

// Two-input element-wise ops
@group(0) @binding(3) var<storage, read> input_b: array<f32>;

@compute @workgroup_size(256)
fn add(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  if (idx >= params.size) { return; }
  output[idx] = input_a[idx] + input_b[idx];
}

@compute @workgroup_size(256)
fn mul(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  if (idx >= params.size) { return; }
  output[idx] = input_a[idx] * input_b[idx];
}

@compute @workgroup_size(256)
fn sub(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  if (idx >= params.size) { return; }
  output[idx] = input_a[idx] - input_b[idx];
}

// Fused add + scale: output = a + scalar * b
@compute @workgroup_size(256)
fn add_scaled(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  if (idx >= params.size) { return; }
  output[idx] = input_a[idx] + params.scalar * input_b[idx];
}

// Softcap logits: softcap * tanh(x / softcap)
@compute @workgroup_size(256)
fn softcap(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  if (idx >= params.size) { return; }
  let cap = params.scalar;
  output[idx] = cap * tanh(input_a[idx] / cap);
}
