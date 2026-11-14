import math

import torch
from torch import nn
from torch.nn import functional as F


def norm(x):
    return F.rms_norm(x, (x.shape[-1],), eps=1e-5)


def rotary(x, cos, sin):
    first, second = x.chunk(2, dim=-1)
    return torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)


class Attention(nn.Module):
    def __init__(self, width, heads, kv_heads, gate):
        super().__init__()
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = width // heads
        self.q = nn.Linear(width, width, bias=False)
        self.k = nn.Linear(width, kv_heads * self.head_dim, bias=False)
        self.v = nn.Linear(width, kv_heads * self.head_dim, bias=False)
        self.out = nn.Linear(width, width, bias=False)
        self.gate = nn.Linear(width, heads, bias=False) if gate else None

    def forward(self, x, cos, sin, probe=False):
        batch, time, width = x.shape
        q = self.q(x).view(batch, time, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(batch, time, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(batch, time, self.kv_heads, self.head_dim).transpose(1, 2)
        q = norm(rotary(q, cos.to(q.dtype), sin.to(q.dtype)))
        k = norm(rotary(k, cos.to(k.dtype), sin.to(k.dtype)))
        k = k.repeat_interleave(self.heads // self.kv_heads, dim=1)
        v = v.repeat_interleave(self.heads // self.kv_heads, dim=1)
        z = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        if self.gate is not None:
            gain = 2 * self.gate(x).transpose(1, 2).unsqueeze(-1).sigmoid()
            z = z * gain
        z = z.transpose(1, 2).contiguous().view(batch, time, width)
        output = self.out(z)
        if not probe:
            return output
        last_query = q[:, :, -1:, :].float()
        logit_max = ((last_query @ k.float().transpose(-2, -1)) / math.sqrt(self.head_dim)).abs().amax()
        return output, z, logit_max


class Block(nn.Module):
    def __init__(self, m, arm):
        super().__init__()
        width = m["width"]
        self.attention = Attention(width, m["heads"], m["kv_heads"], arm.get("gate", False))
        self.up = nn.Linear(width, 2 * m["mlp"], bias=False)
        self.down = nn.Linear(m["mlp"], width, bias=False)
        self.sandwich = arm.get("sandwich", False)

    def write(self, x, cos, sin, probe=False):
        ret = self.attention(norm(x), cos, sin, probe=probe)
        if probe:
            output, z, logit = ret
            return norm(output) if self.sandwich else output, z, logit
        return norm(ret) if self.sandwich else ret

    def forward(self, x, cos, sin):
        x = x + self.write(x, cos, sin)
        a, b = self.up(norm(x)).chunk(2, dim=-1)
        result = self.down(F.silu(a) * b)
        return x + (norm(result) if self.sandwich else result)


class GQA(nn.Module):
    def __init__(self, m, arm):
        super().__init__()
        head_dim = m["width"] // m["heads"]
        if head_dim % 2 or m["width"] % m["heads"] or m["heads"] % m["kv_heads"]:
            raise ValueError("head shapes do not allow rotary grouped-query attention")
        self.embed = nn.Embedding(m["vocab"], m["width"])
        self.blocks = nn.ModuleList(Block(m, arm) for _ in range(m["depth"]))
        inv = 1.0 / (10_000 ** (torch.arange(0, head_dim, 2) / head_dim))
        positions = torch.arange(m["sequence"], dtype=torch.float32)
        angles = torch.outer(positions, inv)[None, None, :, :]
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=0.02)
        for block in self.blocks:
            for project in (block.attention.out, block.down):
                nn.init.normal_(project.weight, std=0.02 / math.sqrt(2 * m["depth"]))
            if block.attention.gate is not None:
                nn.init.zeros_(block.attention.gate.weight)

    def forward(self, tokens, targets, capture=False):
        cos, sin = self.cos[:, :, :tokens.size(1)], self.sin[:, :, :tokens.size(1)]
        x = self.embed(tokens)
        states = []
        for block in self.blocks:
            if capture:
                states.append(x)
            x = block(x, cos, sin)
        logits = F.linear(norm(x), self.embed.weight)
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return (loss, states) if capture else loss
