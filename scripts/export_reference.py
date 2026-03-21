import argparse
import json
import math
import os
from collections import OrderedDict

import torch
import torch.nn.functional as F


def has_value_embedding(layer_idx: int, n_layer: int) -> bool:
    return layer_idx % 2 == (n_layer - 1) % 2


def compute_window_sizes(sequence_len: int, n_layer: int, window_pattern: str) -> list[tuple[int, int]]:
    pattern = window_pattern.upper()
    if not pattern:
        raise ValueError("window_pattern must not be empty")
    long_window = sequence_len
    short_window = long_window // 2
    char_to_window = {
        "L": (long_window, 0),
        "S": (short_window, 0),
    }
    windows: list[tuple[int, int]] = []
    for layer_idx in range(n_layer):
        char = pattern[layer_idx % len(pattern)]
        if char not in char_to_window:
            raise ValueError(f"unsupported window pattern character: {char}")
        windows.append(char_to_window[char])
    if windows:
        windows[-1] = (long_window, 0)
    return windows


def rms_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(variance + eps)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    _, t, _, head_dim = x.shape
    half_dim = head_dim // 2
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]
    cos = cos[:t].view(1, t, 1, half_dim)
    sin = sin[:t].view(1, t, 1, half_dim)
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], dim=-1)


def build_attention_mask(seq_len: int, window_size: tuple[int, int], device: torch.device) -> torch.Tensor:
    left, right = window_size
    indices = torch.arange(seq_len, device=device)
    q = indices.view(seq_len, 1)
    k = indices.view(1, seq_len)
    allowed = (k <= q + right) & (k >= q - left)
    mask = torch.zeros((seq_len, seq_len), dtype=torch.float32, device=device)
    mask.masked_fill_(~allowed, -1e9)
    return mask


def export_to_bin(tensors: OrderedDict[str, torch.Tensor], meta_path: str, bin_path: str) -> None:
    metadata: dict[str, dict[str, int | list[int]]] = {}
    offset = 0
    with open(bin_path, "wb") as handle:
        for name, tensor in tensors.items():
            array = tensor.detach().cpu().float().contiguous().numpy()
            byte_length = int(array.nbytes)
            metadata[name] = {
                "shape": list(array.shape),
                "offset": offset,
                "byteLength": byte_length,
            }
            handle.write(array.tobytes(order="C"))
            offset += byte_length
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")


class ReferenceModel(torch.nn.Module):
    VE_GATE_CHANNELS = 32

    def __init__(self, config: dict[str, int | str]):
        super().__init__()
        self.config = config
        self.vocab_size = int(config["vocabSize"])
        self.n_layer = int(config["nLayer"])
        self.n_head = int(config["nHead"])
        self.n_kv_head = int(config["nKvHead"])
        self.n_embd = int(config["nEmbd"])
        self.sequence_len = int(config["sequenceLen"])
        self.window_pattern = str(config["windowPattern"])
        self.head_dim = self.n_embd // self.n_head
        self.kv_dim = self.n_kv_head * self.head_dim
        self.kv_group_size = self.n_head // self.n_kv_head
        self.window_sizes = compute_window_sizes(self.sequence_len, self.n_layer, self.window_pattern)

        self.wte = torch.nn.Parameter(torch.empty(self.vocab_size, self.n_embd))
        self.lm_head = torch.nn.Parameter(torch.empty(self.vocab_size, self.n_embd))
        self.resid_lambdas = torch.nn.Parameter(torch.ones(self.n_layer))
        self.x0_lambdas = torch.nn.Parameter(torch.full((self.n_layer,), 0.1))

        self.q_weights = torch.nn.ParameterList()
        self.k_weights = torch.nn.ParameterList()
        self.v_weights = torch.nn.ParameterList()
        self.proj_weights = torch.nn.ParameterList()
        self.fc_weights = torch.nn.ParameterList()
        self.mlp_proj_weights = torch.nn.ParameterList()
        self.value_embed_weights = torch.nn.ParameterDict()
        self.ve_gate_weights = torch.nn.ParameterDict()

        for layer_idx in range(self.n_layer):
            self.q_weights.append(torch.nn.Parameter(torch.empty(self.n_embd, self.n_embd)))
            self.k_weights.append(torch.nn.Parameter(torch.empty(self.n_embd, self.kv_dim)))
            self.v_weights.append(torch.nn.Parameter(torch.empty(self.n_embd, self.kv_dim)))
            self.proj_weights.append(torch.nn.Parameter(torch.empty(self.n_embd, self.n_embd)))
            self.fc_weights.append(torch.nn.Parameter(torch.empty(self.n_embd, 4 * self.n_embd)))
            self.mlp_proj_weights.append(torch.nn.Parameter(torch.empty(4 * self.n_embd, self.n_embd)))
            if has_value_embedding(layer_idx, self.n_layer):
                self.value_embed_weights[str(layer_idx)] = torch.nn.Parameter(torch.empty(self.vocab_size, self.kv_dim))
                self.ve_gate_weights[str(layer_idx)] = torch.nn.Parameter(
                    torch.empty(self.VE_GATE_CHANNELS, self.n_kv_head)
                )

        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))
        t = torch.arange(0, self.sequence_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos_table", torch.cos(freqs), persistent=False)
        self.register_buffer("sin_table", torch.sin(freqs), persistent=False)

        self.init_weights()

    def init_weights(self) -> None:
        torch.manual_seed(42)
        self.wte.data.normal_(mean=0.0, std=0.02)
        self.lm_head.data.normal_(mean=0.0, std=0.02)
        self.resid_lambdas.data.fill_(1.0)
        self.x0_lambdas.data.fill_(0.1)

        for layer_idx in range(self.n_layer):
            self.q_weights[layer_idx].data.normal_(mean=0.0, std=0.02)
            self.k_weights[layer_idx].data.normal_(mean=0.0, std=0.02)
            self.v_weights[layer_idx].data.normal_(mean=0.0, std=0.02)
            self.proj_weights[layer_idx].data.zero_()
            self.fc_weights[layer_idx].data.normal_(mean=0.0, std=0.02)
            self.mlp_proj_weights[layer_idx].data.zero_()
            if has_value_embedding(layer_idx, self.n_layer):
                self.value_embed_weights[str(layer_idx)].data.normal_(mean=0.0, std=0.02)
                self.ve_gate_weights[str(layer_idx)].data.zero_()

    def named_reference_tensors(self) -> OrderedDict[str, torch.Tensor]:
        tensors: OrderedDict[str, torch.Tensor] = OrderedDict()
        tensors["wte"] = self.wte
        tensors["lmHead"] = self.lm_head
        tensors["residLambdas"] = self.resid_lambdas
        tensors["x0Lambdas"] = self.x0_lambdas
        for layer_idx in range(self.n_layer):
            tensors[f"layer{layer_idx}_q"] = self.q_weights[layer_idx]
            tensors[f"layer{layer_idx}_k"] = self.k_weights[layer_idx]
            tensors[f"layer{layer_idx}_v"] = self.v_weights[layer_idx]
            tensors[f"layer{layer_idx}_proj"] = self.proj_weights[layer_idx]
            tensors[f"layer{layer_idx}_fc"] = self.fc_weights[layer_idx]
            tensors[f"layer{layer_idx}_mlp_proj"] = self.mlp_proj_weights[layer_idx]
            if has_value_embedding(layer_idx, self.n_layer):
                tensors[f"layer{layer_idx}_value_embed"] = self.value_embed_weights[str(layer_idx)]
                tensors[f"layer{layer_idx}_ve_gate"] = self.ve_gate_weights[str(layer_idx)]
        return tensors

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_len = input_ids.shape
        x = self.wte[input_ids]
        x_norm = rms_norm(x)
        x0 = x_norm
        current = x_norm

        for layer_idx in range(self.n_layer):
            lambda_resid = self.resid_lambdas[layer_idx]
            lambda_x0 = self.x0_lambdas[layer_idx]

            scaled = current * lambda_resid + x0 * lambda_x0
            pre_norm = rms_norm(scaled)
            flat_norm = pre_norm.reshape(batch_size * seq_len, self.n_embd)

            q = (flat_norm @ self.q_weights[layer_idx]).view(batch_size, seq_len, self.n_head, self.head_dim)
            k = (flat_norm @ self.k_weights[layer_idx]).view(batch_size, seq_len, self.n_kv_head, self.head_dim)
            v = (flat_norm @ self.v_weights[layer_idx]).view(batch_size, seq_len, self.n_kv_head, self.head_dim)

            if has_value_embedding(layer_idx, self.n_layer):
                value_embed = self.value_embed_weights[str(layer_idx)][input_ids].view(
                    batch_size, seq_len, self.n_kv_head, self.head_dim
                )
                gate_input = pre_norm[..., : self.VE_GATE_CHANNELS]
                gate = 2.0 * torch.sigmoid(
                    gate_input.reshape(batch_size * seq_len, self.VE_GATE_CHANNELS) @ self.ve_gate_weights[str(layer_idx)]
                )
                gate = gate.view(batch_size, seq_len, self.n_kv_head, 1)
                v = v + value_embed * gate

            q = rms_norm(apply_rotary(q, self.cos_table, self.sin_table))
            k = rms_norm(apply_rotary(k, self.cos_table, self.sin_table))

            if self.kv_group_size > 1:
                k = k.unsqueeze(3).expand(batch_size, seq_len, self.n_kv_head, self.kv_group_size, self.head_dim)
                v = v.unsqueeze(3).expand(batch_size, seq_len, self.n_kv_head, self.kv_group_size, self.head_dim)
                k = k.reshape(batch_size, seq_len, self.n_head, self.head_dim)
                v = v.reshape(batch_size, seq_len, self.n_head, self.head_dim)

            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)

            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            mask = build_attention_mask(seq_len, self.window_sizes[layer_idx], scores.device).view(1, 1, seq_len, seq_len)
            probs = F.softmax(scores + mask, dim=-1)
            attn_out = (probs @ v).permute(0, 2, 1, 3).contiguous().view(batch_size * seq_len, self.n_embd)

            projected = (attn_out @ self.proj_weights[layer_idx]).view(batch_size, seq_len, self.n_embd)
            scaled = scaled + projected

            mlp_norm = rms_norm(scaled).reshape(batch_size * seq_len, self.n_embd)
            hidden = F.relu(mlp_norm @ self.fc_weights[layer_idx]).square()
            mlp_out = (hidden @ self.mlp_proj_weights[layer_idx]).view(batch_size, seq_len, self.n_embd)
            current = scaled + mlp_out

        current = rms_norm(current).reshape(batch_size * seq_len, self.n_embd)
        logits = current @ self.lm_head.t()
        softcap = 15.0
        logits = torch.tanh(logits / softcap) * softcap
        logits_3d = logits.view(batch_size, seq_len, self.vocab_size)

        loss = None
        if targets is not None:
            flat_targets = targets.reshape(-1)
            one_hot_targets = F.one_hot(torch.relu(flat_targets), num_classes=self.vocab_size).to(torch.float32)
            probs = F.softmax(logits, dim=-1)
            log_probs = torch.log(probs + 1e-10)
            unreduced_loss = -(one_hot_targets * log_probs).sum(dim=-1)
            valid_mask = (flat_targets != -1).to(torch.float32)
            loss = (unreduced_loss * valid_mask).mean()

        return logits_3d, loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export PyTorch reference tensors for browser parity validation.")
    parser.add_argument("--output-dir", default="public/reference", help="Directory to write the reference bundle into.")
    parser.add_argument("--batch-size", type=int, default=1, help="Reference batch size.")
    parser.add_argument("--seed", type=int, default=100, help="Seed used to sample the reference batch.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = {
        "vocabSize": 8192,
        "nLayer": 4,
        "nHead": 4,
        "nKvHead": 4,
        "nEmbd": 256,
        "sequenceLen": 256,
        "windowPattern": "SSSL",
    }

    os.makedirs(args.output_dir, exist_ok=True)
    model = ReferenceModel(config)
    weights = model.named_reference_tensors()
    export_to_bin(weights, os.path.join(args.output_dir, "weights_meta.json"), os.path.join(args.output_dir, "weights.bin"))

    torch.manual_seed(args.seed)
    inputs = torch.randint(0, int(config["vocabSize"]), (args.batch_size, int(config["sequenceLen"])), dtype=torch.int64)
    targets = torch.randint(0, int(config["vocabSize"]), (args.batch_size, int(config["sequenceLen"])), dtype=torch.int64)

    with open(os.path.join(args.output_dir, "batch.json"), "w", encoding="utf-8") as handle:
        json.dump({"inputs": inputs.tolist(), "targets": targets.tolist()}, handle, indent=2)
        handle.write("\n")

    logits, loss = model(inputs, targets)
    if loss is None:
        raise RuntimeError("reference loss was not produced")

    with open(os.path.join(args.output_dir, "logits.bin"), "wb") as handle:
        handle.write(logits.detach().cpu().float().contiguous().numpy().tobytes(order="C"))

    loss.backward()

    with open(os.path.join(args.output_dir, "loss.json"), "w", encoding="utf-8") as handle:
        json.dump({"loss": float(loss.item())}, handle, indent=2)
        handle.write("\n")

    grads: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, tensor in weights.items():
        if tensor.grad is None:
            raise RuntimeError(f"missing gradient for {name}")
        grads[name] = tensor.grad
    export_to_bin(grads, os.path.join(args.output_dir, "grads_meta.json"), os.path.join(args.output_dir, "grads.bin"))

    print(f"Exported reference bundle to {args.output_dir}")
    print(f"Reference loss: {loss.item():.8f}")


if __name__ == "__main__":
    main()
