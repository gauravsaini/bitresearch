// Cross-entropy loss: -log(softmax(logits)[target]) per token
// logits: [B*T, V], targets: [B*T], output: scalar or per-token losses

struct CEParams {
  N: u32,       // batch*seq (number of tokens)
  V: u32,       // vocab size
  ignore_idx: i32,  // index to ignore (-1 typically)
  _pad: u32,
}

@group(0) @binding(0) var<storage, read> logits: array<f32>;     // [N, V]
@group(0) @binding(1) var<storage, read> targets: array<i32>;    // [N]
@group(0) @binding(2) var<storage, read_write> losses: array<f32>;  // [N]
@group(0) @binding(3) var<uniform> params: CEParams;

@compute @workgroup_size(1)
fn cross_entropy_forward(@builtin(global_invocation_id) gid: vec3u) {
  let n = gid.x;
  if (n >= params.N) { return; }

  let target = targets[n];
  if (target == params.ignore_idx) {
    losses[n] = 0.0;
    return;
  }

  let offset = n * params.V;

  // Find max logit for numerical stability
  var max_logit: f32 = logits[offset];
  for (var v: u32 = 1u; v < params.V; v++) {
    max_logit = max(max_logit, logits[offset + v]);
  }

  // Compute log-sum-exp
  var sum_exp: f32 = 0.0;
  for (var v: u32 = 0u; v < params.V; v++) {
    sum_exp += exp(logits[offset + v] - max_logit);
  }
  let log_sum_exp = log(sum_exp) + max_logit;

  // Loss = log_sum_exp - logits[target]
  losses[n] = log_sum_exp - logits[offset + u32(target)];
}

// Backward: dlogits = softmax(logits) - one_hot(target)
@group(0) @binding(4) var<storage, read_write> grad_logits: array<f32>;  // [N, V]
@group(0) @binding(5) var<storage, read> grad_loss: array<f32>;          // [N] (upstream gradient, 1/N for mean)

@compute @workgroup_size(1)
fn cross_entropy_backward(@builtin(global_invocation_id) gid: vec3u) {
  let n = gid.x;
  if (n >= params.N) { return; }

  let target = targets[n];
  if (target == params.ignore_idx) {
    // Zero out gradients for ignored positions
    let offset = n * params.V;
    for (var v: u32 = 0u; v < params.V; v++) {
      grad_logits[offset + v] = 0.0;
    }
    return;
  }

  let offset = n * params.V;
  let upstream = grad_loss[n];

  // Find max logit
  var max_logit: f32 = logits[offset];
  for (var v: u32 = 1u; v < params.V; v++) {
    max_logit = max(max_logit, logits[offset + v]);
  }

  // Compute softmax
  var sum_exp: f32 = 0.0;
  for (var v: u32 = 0u; v < params.V; v++) {
    sum_exp += exp(logits[offset + v] - max_logit);
  }
  let inv_sum = 1.0 / sum_exp;

  // grad = upstream * (softmax - one_hot)
  for (var v: u32 = 0u; v < params.V; v++) {
    var prob = exp(logits[offset + v] - max_logit) * inv_sum;
    if (v == u32(target)) {
      prob -= 1.0;
    }
    grad_logits[offset + v] = upstream * prob;
  }
}

// Reduction: compute mean of per-token losses
struct ReduceParams {
  N: u32,
  count: u32,   // number of non-ignored tokens
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<storage, read> reduce_input: array<f32>;
@group(0) @binding(1) var<storage, read_write> reduce_output: array<f32>;  // [1]
@group(0) @binding(2) var<uniform> reduce_params: ReduceParams;

@compute @workgroup_size(1)
fn reduce_mean(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x != 0u) { return; }
  var sum: f32 = 0.0;
  for (var i: u32 = 0u; i < reduce_params.N; i++) {
    sum += reduce_input[i];
  }
  reduce_output[0] = sum / f32(reduce_params.count);
}
