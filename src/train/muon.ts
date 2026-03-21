/**
 * MuonAdamW Optimizer — TensorFlow.js port of Karpathy's autoresearch optimizer.
 *
 * Combined optimizer:
 *   - Muon for 2D matrix params (Polar Express orthogonalization + NorMuon variance reduction)
 *   - AdamW for 1D/bias/embedding params
 *
 * Reference: https://github.com/karpathy/autoresearch/blob/master/train.py
 */

import * as tf from '@tensorflow/tfjs';

// ---------------------------------------------------------------------------
// Polar Express Newton-Schulz coefficients
// ---------------------------------------------------------------------------
// Karpathy's 5-coefficient set for 5 iterations of Newton-Schulz
// Each triple (a, b, c) defines one step: A = X^T @ X, B = b*A + c*A@A, X = a*X + X@B
const POLAR_EXPRESS_COEFFS: [number, number, number][] = [
  [8.156554524902461, -22.48329292557795, 15.878769915207462],
  [4.042929935166739, -2.808917465908714, 0.5000178451051316],
  [3.8916678022926607, -2.772484153217685, 0.5060648178503393],
  [3.285753657755655, -2.3681294933425376, 0.46449024233003106],
  [2.3465413258596377, -1.7097828382687081, 0.42323551169305323],
];

// ---------------------------------------------------------------------------
// Interfaces
// ---------------------------------------------------------------------------

export interface MuonParamGroup {
  kind: 'muon' | 'adamw';
  params: tf.Variable[];
  lr: number;
  weight_decay: number;
  // Muon-specific
  momentum?: number;
  ns_steps?: number;
  beta2?: number;
  // AdamW-specific
  betas?: [number, number];
  eps?: number;
}

export interface MuonAdamWConfig {
  // Per-group LR (before dmodel scaling)
  unembedding_lr: number;
  embedding_lr: number;
  matrix_lr: number;
  scalar_lr: number;
  weight_decay: number;
  adam_betas: [number, number];
  // Model dimension for LR scaling
  model_dim: number;
}

// ---------------------------------------------------------------------------
// AdamW Step (single parameter)
// ---------------------------------------------------------------------------

function adamWStep(
  param: tf.Variable,
  grad: tf.Tensor,
  expAvg: tf.Variable,
  expAvgSq: tf.Variable,
  step: number,
  lr: number,
  beta1: number,
  beta2: number,
  eps: number,
  wd: number
): void {
  tf.tidy(() => {
    // Decoupled weight decay
    const decayed = param.mul(1 - lr * wd);

    // EMA updates
    const newExpAvg = expAvg.mul(beta1).add(grad.mul(1 - beta1));
    const newExpAvgSq = expAvgSq.mul(beta2).add(grad.square().mul(1 - beta2));

    // Bias correction
    const bias1 = 1 - Math.pow(beta1, step);
    const bias2 = 1 - Math.pow(beta2, step);

    // Adam step
    const denom = newExpAvgSq.div(bias2).sqrt().add(eps);
    const stepSize = lr / bias1;
    const update = newExpAvg.div(bias2 > 0 ? denom : tf.scalar(1)).mul(stepSize);

    const newParam = decayed.sub(update);

    // Apply
    param.assign(newParam);
    expAvg.assign(newExpAvg);
    expAvgSq.assign(newExpAvgSq);
  });
}

// ---------------------------------------------------------------------------
// Polar Express Newton-Schulz Orthogonalization
// ---------------------------------------------------------------------------

function polarExpress(X: tf.Tensor2D, nsSteps: number): tf.Tensor2D {
  return tf.tidy(() => {
    const [rows, cols] = X.shape;
    const largerFirst = rows >= cols;

    // Normalize: X = X / (||X|| * 1.02 + 1e-6)
    const norm = X.norm();
    X = X.div(norm.mul(1.02).add(1e-6));
    X = X.asType('float32');

    for (let i = 0; i < Math.min(nsSteps, POLAR_EXPRESS_COEFFS.length); i++) {
      const [a, b, c] = POLAR_EXPRESS_COEFFS[i];

      if (largerFirst) {
        // A = X^T @ X, B = b*A + c*(A@A), X = a*X + X@B
        const A = tf.matMul(X, X, true, false) as tf.Tensor2D; // X^T @ X
        const AA = tf.matMul(A, A) as tf.Tensor2D;
        const B = tf.add(A.mul(b), AA.mul(c)) as tf.Tensor2D;
        X = tf.add(X.mul(a), tf.matMul(X, B)) as tf.Tensor2D;
      } else {
        // A = X @ X^T, B = b*A + c*(A@A), X = a*X + B@X
        const A = tf.matMul(X, X, false, true) as tf.Tensor2D; // X @ X^T
        const AA = tf.matMul(A, A) as tf.Tensor2D;
        const B = tf.add(A.mul(b), AA.mul(c)) as tf.Tensor2D;
        X = tf.add(X.mul(a), tf.matMul(B, X)) as tf.Tensor2D;
      }
    }

    return X;
  });
}

// ---------------------------------------------------------------------------
// Muon Step (single parameter — a 2D matrix)
// ---------------------------------------------------------------------------

function muonStep(
  param: tf.Variable,
  grad: tf.Tensor,
  momentumBuffer: tf.Variable,
  secondMomentumBuffer: tf.Variable,
  momentum: number,
  lr: number,
  wd: number,
  beta2: number,
  nsSteps: number
): void {
  tf.tidy(() => {
    // Nesterov momentum
    const newMomentumBuf = momentumBuffer.mul(momentum).add(grad.mul(1 - momentum));
    const nesterovGrad = grad.mul(1 - momentum).add(newMomentumBuf.mul(momentum));

    // Polar Express orthogonalization
    let g = polarExpress(nesterovGrad.asType('float32') as tf.Tensor2D, nsSteps);

    // NorMuon variance reduction
    const [rows, cols] = g.shape;
    const largerFirst = rows >= cols;
    const redDim = largerFirst ? 1 : 0;
    const redDimSize = largerFirst ? cols : rows;

    // v_mean = mean(g^2) along reduction dim, keep shape
    const vMean = g.square().mean(redDim, true);

    // v_norm_sq = sum(v_mean) * redDimSize
    const vNormSq = vMean.sum().mul(redDimSize);
    const vNorm = vNormSq.sqrt();

    // Update second momentum buffer (EMA of vMean)
    const newSecondBuf = secondMomentumBuffer.mul(beta2).add(vMean.mul(1 - beta2));

    // step_size = rsqrt(second_momentum_buffer + 1e-10)
    const stepSize = newSecondBuf.add(1e-10).rsqrt();

    // scaled_sq_sum = v_mean * redDimSize * step_size^2
    const scaledSqSum = vMean.mul(redDimSize).mul(stepSize.square());
    const vNormNew = scaledSqSum.sum().sqrt();

    // final_scale = step_size * (v_norm / v_norm_new)
    const finalScale = stepSize.mul(vNorm.div(vNormNew.add(1e-10)));

    g = g.mul(finalScale);

    // Cautious weight decay: only apply WD where (grad * param) >= 0
    const mask = tf.cast(grad.mul(param).greaterEqual(0), 'float32');
    const wdTerm = param.mul(wd).mul(mask);

    // Update: param -= lr * (g + wd * param * mask)
    const newParam = param.sub(g.mul(lr).add(wdTerm.mul(lr)));

    param.assign(newParam);
    momentumBuffer.assign(newMomentumBuf);
    secondMomentumBuffer.assign(newSecondBuf);
  });
}

// ---------------------------------------------------------------------------
// MuonAdamW Optimizer Class
// ---------------------------------------------------------------------------

export class MuonAdamWOptimizer {
  private config: MuonAdamWConfig;
  private paramGroups: MuonParamGroup[] = [];

  // AdamW state: per-variable
  private adamState: Map<string, {
    step: number;
    expAvg: tf.Variable;
    expAvgSq: tf.Variable;
  }> = new Map();

  // Muon state: per-variable
  private muonState: Map<string, {
    step: number;
    momentumBuffer: tf.Variable;
    secondMomentumBuffer: tf.Variable;
  }> = new Map();

  // Schedules (set externally)
  lrMultiplier: number = 1.0;
  muonMomentum: number = 0.95;
  muonWeightDecay: number = 0.0;

  constructor(config: MuonAdamWConfig, paramGroups: {
    lmHead: tf.Variable[];
    wte: tf.Variable[];
    valueEmbeds: tf.Variable[];
    scalars: tf.Variable[];
    matrices: tf.Variable[];
  }) {
    this.config = config;

    const dmodelLrScale = Math.pow(config.model_dim / 768, -0.5);
    console.log(`[MuonAdamW] Scaling LRs by (768/${config.model_dim})^0.5 = ${dmodelLrScale.toFixed(4)}`);

    // Build parameter groups (Karpathy's exact grouping)
    this.paramGroups = [
      {
        kind: 'adamw',
        params: paramGroups.lmHead,
        lr: config.unembedding_lr * dmodelLrScale,
        weight_decay: 0.0,
        betas: config.adam_betas,
        eps: 1e-10,
      },
      {
        kind: 'adamw',
        params: paramGroups.wte,
        lr: config.embedding_lr * dmodelLrScale,
        weight_decay: 0.0,
        betas: config.adam_betas,
        eps: 1e-10,
      },
      {
        kind: 'adamw',
        params: paramGroups.valueEmbeds,
        lr: config.embedding_lr * dmodelLrScale,
        weight_decay: 0.0,
        betas: config.adam_betas,
        eps: 1e-10,
      },
      {
        kind: 'adamw',
        // residLambdas only (first scalar)
        params: paramGroups.scalars.slice(0, 1),
        lr: config.scalar_lr * 0.01,
        weight_decay: 0.0,
        betas: config.adam_betas,
        eps: 1e-10,
      },
      {
        kind: 'adamw',
        // x0Lambdas only (second scalar) — Karpathy uses betas=(0.96, 0.95)
        params: paramGroups.scalars.slice(1),
        lr: config.scalar_lr,
        weight_decay: 0.0,
        betas: [0.96, 0.95] as [number, number],
        eps: 1e-10,
      },
      {
        kind: 'muon',
        params: paramGroups.matrices,
        lr: config.matrix_lr,
        weight_decay: config.weight_decay,
        momentum: 0.95,
        ns_steps: 5,
        beta2: 0.95,
      },
    ];

    // Initialize state for all variables
    for (const group of this.paramGroups) {
      for (const p of group.params) {
        if (group.kind === 'adamw') {
          if (!this.adamState.has(p.name)) {
            this.adamState.set(p.name, {
              step: 0,
              expAvg: tf.variable(tf.zerosLike(p), false, `adam_m_${p.name}`),
              expAvgSq: tf.variable(tf.zerosLike(p), false, `adam_v_${p.name}`),
            });
          }
        } else if (group.kind === 'muon') {
          if (!this.muonState.has(p.name)) {
            const shape = p.shape;
            const stateShape = shape.length >= 2
              ? (shape[shape.length - 2] >= shape[shape.length - 1]
                  ? [shape.length, shape[shape.length - 2], 1]
                  : [shape.length, 1, shape[shape.length - 1]])
              : shape;

            this.muonState.set(p.name, {
              step: 0,
              momentumBuffer: tf.variable(tf.zerosLike(p), false, `muon_mom_${p.name}`),
              secondMomentumBuffer: tf.variable(tf.zeros(stateShape), false, `muon_sec_${p.name}`),
            });
          }
        }
      }
    }
  }

  /**
   * Apply gradients using the collected gradient dictionary.
   * This is called after tf.variableGrads() returns.
   */
  step(grads: Record<string, tf.Tensor>): void {
    for (const group of this.paramGroups) {
      if (group.kind === 'adamw') {
        this.stepAdamW(group, grads);
      } else if (group.kind === 'muon') {
        this.stepMuon(group, grads);
      }
    }
  }

  private stepAdamW(group: MuonParamGroup, grads: Record<string, tf.Tensor>): void {
    const lr = group.lr! * this.lrMultiplier;
    const [beta1, beta2] = group.betas!;
    const eps = group.eps!;
    const wd = group.weight_decay;

    for (const p of group.params) {
      const grad = grads[p.name];
      if (!grad) continue;

      const state = this.adamState.get(p.name);
      if (!state) continue;

      state.step++;
      adamWStep(p, grad, state.expAvg, state.expAvgSq,
        state.step, lr, beta1, beta2, eps, wd);
    }
  }

  private stepMuon(group: MuonParamGroup, grads: Record<string, tf.Tensor>): void {
    const lr = group.lr! * this.lrMultiplier;
    const wd = this.muonWeightDecay;
    const momentum = this.muonMomentum;
    const beta2 = group.beta2!;
    const nsSteps = group.ns_steps!;

    // Scale LR by sqrt(max(rows/cols, cols/rows)) per Karpathy
    for (const p of group.params) {
      const grad = grads[p.name];
      if (!grad) continue;

      const state = this.muonState.get(p.name);
      if (!state) continue;

      state.step++;

      // LR scale: max(1.0, sqrt(max(rows/cols, cols/rows)))
      const shape = p.shape;
      if (shape.length >= 2) {
        const lrScale = Math.sqrt(Math.max(shape[shape.length - 2] / shape[shape.length - 1],
                                            shape[shape.length - 1] / shape[shape.length - 2]));
        const scaledLr = lr * Math.max(1.0, lrScale);

        muonStep(p, grad, state.momentumBuffer, state.secondMomentumBuffer,
          momentum, scaledLr, wd, beta2, nsSteps);
      }
    }
  }

  /**
   * Get optimizer state for checkpointing.
   */
  getState(): Record<string, any> {
    const state: Record<string, any> = {};
    for (const [name, s] of this.adamState) {
      state[`adam_${name}_step`] = s.step;
      state[`adam_${name}_m`] = s.expAvg;
      state[`adam_${name}_v`] = s.expAvgSq;
    }
    for (const [name, s] of this.muonState) {
      state[`muon_${name}_step`] = s.step;
      state[`muon_${name}_mom`] = s.momentumBuffer;
      state[`muon_${name}_sec`] = s.secondMomentumBuffer;
    }
    return state;
  }

  dispose(): void {
    for (const s of this.adamState.values()) {
      s.expAvg.dispose();
      s.expAvgSq.dispose();
    }
    for (const s of this.muonState.values()) {
      s.momentumBuffer.dispose();
      s.secondMomentumBuffer.dispose();
    }
    this.adamState.clear();
    this.muonState.clear();
  }
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

export function createMuonAdamW(
  model: {
    getParamGroups(): {
      lmHead: tf.Variable[];
      wte: tf.Variable[];
      valueEmbeds: tf.Variable[];
      scalars: tf.Variable[];
      matrices: tf.Variable[];
    };
    config: { nEmbd: number };
  },
  overrides: Partial<MuonAdamWConfig> = {}
): MuonAdamWOptimizer {
  const defaults: MuonAdamWConfig = {
    unembedding_lr: 0.004,
    embedding_lr: 0.6,
    matrix_lr: 0.04,
    scalar_lr: 0.5,
    weight_decay: 0.2,
    adam_betas: [0.8, 0.95],
    model_dim: model.config.nEmbd,
  };
  const config = { ...defaults, ...overrides };
  return new MuonAdamWOptimizer(config, model.getParamGroups());
}
