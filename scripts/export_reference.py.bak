import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import json
import os

class RMSNorm(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps)

def apply_rotary(x, cos, sin):
    B, T, nh, hd = x.shape
    x1, x2 = x.split(hd // 2, dim=-1)
    half_rotated = torch.cat([-x2, x1], dim=-1)
    cos = cos.view(1, T, 1, hd)
    sin = sin.view(1, T, 1, hd)
    return x * cos + half_rotated * sin

class GPTLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embd = config['nEmbd']
        self.n_head = config['nHead']
        self.head_dim = self.n_embd // self.n_head
        
        self.qWeight = nn.Parameter(torch.empty(self.n_embd, self.n_embd))
        self.kWeight = nn.Parameter(torch.empty(self.n_embd, self.n_embd))
        self.vWeight = nn.Parameter(torch.empty(self.n_embd, self.n_embd))
        self.projWeight = nn.Parameter(torch.empty(self.n_embd, self.n_embd))
        
        self.fcWeight = nn.Parameter(torch.empty(self.n_embd, 4 * self.n_embd))
        self.mlpProjWeight = nn.Parameter(torch.empty(4 * self.n_embd, self.n_embd))
        
        self.rmsnorm = RMSNorm()
        
    def forward(self, x, x0, lambdaResid, lambdaX0, cos, sin):
        B, T, C = x.shape
        scaled = x * lambdaResid + x0 * lambdaX0
        preNorm = self.rmsnorm(scaled)
        
        q = torch.matmul(preNorm, self.qWeight).view(B, T, self.n_head, self.head_dim)
        k = torch.matmul(preNorm, self.kWeight).view(B, T, self.n_head, self.head_dim)
        v = torch.matmul(preNorm, self.vWeight).view(B, T, self.n_head, self.head_dim)
        
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        
        q = self.rmsnorm(q).transpose(1, 2)
        k = self.rmsnorm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        scores = scores.masked_fill(mask == 0, -1e9)
        
        probs = F.softmax(scores, dim=-1)
        attnOut = torch.matmul(probs, v).transpose(1, 2).contiguous().view(B, T, C)
        
        projected = torch.matmul(attnOut, self.projWeight)
        scaled = scaled + projected
        
        mlpNorm = self.rmsnorm(scaled)
        hidden = torch.matmul(mlpNorm, self.fcWeight)
        hidden = F.relu(hidden) ** 2
        mlpOut = torch.matmul(hidden, self.mlpProjWeight)
        
        current = scaled + mlpOut
        return current

class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config['vocabSize']
        self.n_embd = config['nEmbd']
        self.n_layer = config['nLayer']
        self.seq_len = config['sequenceLen']
        
        self.wte = nn.Parameter(torch.empty(self.vocab_size, self.n_embd))
        self.lmHead = nn.Parameter(torch.empty(self.vocab_size, self.n_embd))
        
        self.residLambdas = nn.Parameter(torch.ones(self.n_layer))
        self.x0Lambdas = nn.Parameter(torch.full((self.n_layer,), 0.1))
        
        self.layers = nn.ModuleList([GPTLayer(config) for _ in range(self.n_layer)])
        self.rmsnorm = RMSNorm()
        
        hd = self.n_embd // config['nHead']
        inv_freq = 1.0 / (10000 ** (torch.arange(0, hd, 2).float() / hd))
        t = torch.arange(self.seq_len).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cosTable', emb.cos())
        self.register_buffer('sinTable', emb.sin())
        
        self._init_weights()
        
    def _init_weights(self):
        torch.manual_seed(42)
        std = 0.02
        projStd = 0.02 / math.sqrt(2 * self.n_layer)
        
        nn.init.normal_(self.wte, mean=0.0, std=std)
        nn.init.normal_(self.lmHead, mean=0.0, std=std)
        for layer in self.layers:
            nn.init.normal_(layer.qWeight, std=std)
            nn.init.normal_(layer.kWeight, std=std)
            nn.init.normal_(layer.vWeight, std=std)
            nn.init.normal_(layer.projWeight, std=projStd)
            nn.init.normal_(layer.fcWeight, std=std)
            nn.init.normal_(layer.mlpProjWeight, std=projStd)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.wte[idx]
        xNorm = self.rmsnorm(x)
        
        x0 = xNorm
        current = xNorm
        
        for i, layer in enumerate(self.layers):
            current = layer(current, x0, self.residLambdas[i], self.x0Lambdas[i], self.cosTable[:T], self.sinTable[:T])
            
        current = self.rmsnorm(current)
        logits = torch.matmul(current, self.lmHead.t())
        
        softcap = 15.0
        logits = torch.tanh(logits / softcap) * softcap
        
        loss = None
        if targets is not None:
            logits_flat = logits.view(-1, self.vocab_size)
            targets_flat = targets.view(-1)
            
            valid_mask = targets_flat != -1
            # Prevent -1 index error in cross_entropy
            targets_flat_zeroed = targets_flat.clone()
            targets_flat_zeroed[~valid_mask] = 0
            
            # Match TFJS tf.losses.softmaxCrossEntropy with Reduction.MEAN behavior
            loss_unreduced = F.cross_entropy(logits_flat, targets_flat_zeroed, reduction='none')
            loss = (loss_unreduced * valid_mask.float()).mean()
            
        return logits, loss

def export_to_bin(tensor_dict, meta_path, bin_path):
    metadata = {}
    with open(bin_path, 'wb') as f:
        offset = 0
        for name, tensor in tensor_dict.items():
            np_arr = tensor.detach().cpu().numpy().astype(np.float32).copy(order='C')
            shape = list(np_arr.shape)
            byte_size = np_arr.nbytes
            
            metadata[name] = {
                'shape': shape,
                'offset': offset,
                'byteLength': byte_size
            }
            
            f.write(np_arr.tobytes())
            offset += byte_size
            
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    config = {
        'vocabSize': 8192,
        'nLayer': 4,
        'nHead': 4,
        'nKvHead': 4,
        'nEmbd': 256,
        'sequenceLen': 256
    }
    
    # Create the target directory inside public so we can fetch it from Vite
    target_dir = 'public/reference'
    os.makedirs(target_dir, exist_ok=True)
    model = GPTModel(config)
    
    # 1. Export initialized weights
    weights = {}
    weights["wte"] = model.wte
    weights["lmHead"] = model.lmHead
    weights["residLambdas"] = model.residLambdas
    weights["x0Lambdas"] = model.x0Lambdas
    for i, layer in enumerate(model.layers):
        weights[f"layer{i}_q"] = layer.qWeight
        weights[f"layer{i}_k"] = layer.kWeight
        weights[f"layer{i}_v"] = layer.vWeight
        weights[f"layer{i}_proj"] = layer.projWeight
        weights[f"layer{i}_fc"] = layer.fcWeight
        weights[f"layer{i}_mlp_proj"] = layer.mlpProjWeight
        
    export_to_bin(weights, f'{target_dir}/weights_meta.json', f'{target_dir}/weights.bin')
    
    # 2. Run forward pass with seeded batch
    torch.manual_seed(100)
    batch_size = 2
    inputs = torch.randint(0, config['vocabSize'], (batch_size, config['sequenceLen']))
    targets = torch.randint(-1, config['vocabSize'], (batch_size, config['sequenceLen']))
    
    # Save inputs/targets so TFJS can use the EXACT same batch
    with open(f'{target_dir}/batch.json', 'w') as f:
        json.dump({
            'inputs': inputs.tolist(),
            'targets': targets.tolist()
        }, f)
        
    logits, loss = model(inputs, targets)
    
    # Save exact logits to isolate forward pass
    logits_np = logits.detach().cpu().numpy().astype(np.float32).copy(order='C')
    with open(f'{target_dir}/logits.bin', 'wb') as f:
        f.write(logits_np.tobytes())
        
    loss.backward()
    
    with open(f'{target_dir}/loss.json', 'w') as f:
        json.dump({'loss': loss.item()}, f)
    
    print(f"PyTorch Loss: {loss.item():.8f}")
    
    # 3. Export Gradients
    grads = {}
    for name, tensor in weights.items():
        if tensor.grad is not None:
            grads[name] = tensor.grad
            
    export_to_bin(grads, f'{target_dir}/grads_meta.json', f'{target_dir}/grads.bin')
    print("Export complete!")
