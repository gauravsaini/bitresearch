// Matrix multiplication: C = A @ B
// A: [M, K], B: [K, N], C: [M, N]
// Uses tiled approach for better cache utilization

struct Params {
  M: u32,
  K: u32,
  N: u32,
  alpha: f32,  // C = alpha * A @ B + beta * C
  beta: f32,
}

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const TILE_SIZE: u32 = 16u;

var<workgroup> tileA: array<array<f32, 16>, 16>;
var<workgroup> tileB: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16)
fn matmul(@builtin(global_invocation_id) gid: vec3u,
          @builtin(local_invocation_id) lid: vec3u,
          @builtin(workgroup_id) wid: vec3u) {
  let row = gid.y;
  let col = gid.x;
  let localRow = lid.y;
  let localCol = lid.x;

  var sum: f32 = 0.0;
  let numTiles = (params.K + TILE_SIZE - 1u) / TILE_SIZE;

  for (var t: u32 = 0u; t < numTiles; t++) {
    // Load tile of A
    let aCol = t * TILE_SIZE + localCol;
    if (row < params.M && aCol < params.K) {
      tileA[localRow][localCol] = A[row * params.K + aCol];
    } else {
      tileA[localRow][localCol] = 0.0;
    }

    // Load tile of B
    let bRow = t * TILE_SIZE + localRow;
    if (bRow < params.K && col < params.N) {
      tileB[localRow][localCol] = B[bRow * params.N + col];
    } else {
      tileB[localRow][localCol] = 0.0;
    }

    workgroupBarrier();

    // Compute partial dot product
    for (var k: u32 = 0u; k < TILE_SIZE; k++) {
      sum += tileA[localRow][k] * tileB[k][localCol];
    }

    workgroupBarrier();
  }

  if (row < params.M && col < params.N) {
    let idx = row * params.N + col;
    C[idx] = params.alpha * sum + params.beta * C[idx];
  }
}

// Batched matmul: C[b] = A[b] @ B[b] for batch dimension
// A: [B, M, K], B: [B, K, N], C: [B, M, N]
struct BatchParams {
  M: u32,
  K: u32,
  N: u32,
  batch: u32,
}

@group(0) @binding(0) var<storage, read> bA: array<f32>;
@group(0) @binding(1) var<storage, read> bB: array<f32>;
@group(0) @binding(2) var<storage, read_write> bC: array<f32>;
@group(0) @binding(3) var<uniform> bparams: BatchParams;

@compute @workgroup_size(16, 16)
fn batched_matmul(@builtin(global_invocation_id) gid: vec3u) {
  let col = gid.x;
  let row = gid.y;
  let batch = gid.z;

  if (batch >= bparams.batch || row >= bparams.M || col >= bparams.N) {
    return;
  }

  let aOffset = batch * bparams.M * bparams.K;
  let bOffset = batch * bparams.K * bparams.N;
  let cOffset = batch * bparams.M * bparams.N;

  var sum: f32 = 0.0;
  for (var k: u32 = 0u; k < bparams.K; k++) {
    sum += bA[aOffset + row * bparams.K + k] * bB[bOffset + k * bparams.N + col];
  }

  bC[cOffset + row * bparams.N + col] = sum;
}
