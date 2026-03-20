import * as tf from '@tensorflow/tfjs';
import { GPTModel } from './gpt';

async function fetchBin(url: string) {
  const res = await fetch(url);
  return await res.arrayBuffer();
}

async function fetchJson(url: string) {
  const res = await fetch(url);
  return await res.json();
}

async function runValidation() {
  const statusEl = document.getElementById('status')!;
  const logsEl = document.getElementById('logs')!;
  
  const log = (msg: string) => {
    console.log(msg);
    logsEl.innerHTML += `<div>${msg}</div>`;
  };

  statusEl.innerText = 'Loading reference data...';
  
  const weightsMeta = await fetchJson('/reference/weights_meta.json');
  const weightsBin = await fetchBin('/reference/weights.bin');
  const gradsMeta = await fetchJson('/reference/grads_meta.json');
  const gradsBin = await fetchBin('/reference/grads.bin');
  const logitsBin = await fetchBin('/reference/logits.bin');
  const batch = await fetchJson('/reference/batch.json');
  const refLossData = await fetchJson('/reference/loss.json');

  const config = {
    vocabSize: 8192,
    nLayer: 4,
    nHead: 4,
    nKvHead: 4,
    nEmbd: 256,
    sequenceLen: 256,
    windowPattern: "SSSL"
  };

  statusEl.innerText = 'Initializing WebGPU Model...';
  const model = new GPTModel(config);
  await model.init();

  statusEl.innerText = 'Loading PyTorch Reference Weights...';
  tf.tidy(() => {
    for (const [name, variable] of Object.entries(model.getTrainableVariablesByName())) {
         const meta = weightsMeta[name];
         if (!meta) {
           log(`Missing weight metadata for ${name}`);
           continue;
         }
         const f32 = new Float32Array(weightsBin, meta.offset, meta.byteLength / 4);
         const tensor = tf.tensor(f32, meta.shape);
         variable.assign(tensor);
    }
  });

  statusEl.innerText = 'Running TFJS Forward Pass Parity Check...';
  const inputs = tf.tensor2d(batch.inputs, [2, 256], 'int32');
  const targets = tf.tensor2d(batch.targets, [2, 256], 'int32');
  let fwdLoss: number;
  tf.tidy(() => {
    const fwd = model.forward(inputs, targets, true);
    fwdLoss = fwd.loss.dataSync()[0];
    const tfLogits = fwd.logits!;
    const ptLogitsF32 = new Float32Array(logitsBin);
    const ptLogits = tf.tensor(ptLogitsF32, tfLogits.shape);
    const logitsDiff = tfLogits.sub(ptLogits).abs().max();
    const d = logitsDiff.dataSync()[0];
    log(`<strong>PyTorch Loss:</strong> ${refLossData.loss.toFixed(6)}`);
    log(`<strong>TF.js Loss:   </strong> ${fwdLoss.toFixed(6)}`);
    log(`<strong>Loss Delta:   </strong> ${Math.abs(fwdLoss - refLossData.loss).toFixed(6)}`);
    log(`<strong>Logits Max Diff:</strong> ${d.toFixed(6)}`);
  });

  statusEl.innerText = 'Running TFJS Backward Pass on WebGPU...';

  const { value, grads } = tf.variableGrads(() => {
    return model.forward(inputs, targets).loss;
  });

  const lossValue = await value.data();

  statusEl.innerText = 'Comparing Gradients...';
  let passed = true;
  
  for (const [name, variable] of Object.entries(model.getTrainableVariablesByName())) {
    const tfGrad = grads[variable.name];
    if (!tfGrad) continue;
    
    const meta = gradsMeta[name];
    if (!meta) {
        log(`Missing gradient metadata for ${name}`);
        continue;
    }
    const f32 = new Float32Array(gradsBin, meta.offset, meta.byteLength / 4);
    const ptGrad = tf.tensor(f32, meta.shape);
    
    // Compute max difference and L2 norm difference
    const diff = tfGrad.sub(ptGrad).abs().max();
    const l2A = tf.norm(tfGrad);
    const l2B = tf.norm(ptGrad);
    
    const maxDiff = (await diff.data())[0];
    const nA = (await l2A.data())[0];
    const nB = (await l2B.data())[0];
    
    const color = maxDiff > 1e-4 ? 'red' : 'green';
    log(`<span style="color: ${color}">${name} | maxDiff: ${maxDiff.toFixed(6)} | tf-l2: ${nA.toFixed(6)} | pt-l2: ${nB.toFixed(6)}</span>`);
    
    if (maxDiff > 1e-4) {
      passed = false;
    }
    
    tfGrad.dispose();
    ptGrad.dispose();
  }

  statusEl.innerHTML = passed 
    ? '<span style="color: #4ade80;">✅ Validation Passed! Gradients exactly match PyTorch.</span>' 
    : '<span style="color: #f87171;">❌ Validation Failed! Mismatches found.</span>';
}

runValidation().catch(e => {
  console.error(e);
  document.getElementById('status')!.innerText = `Error: ${e.message}`;
});
