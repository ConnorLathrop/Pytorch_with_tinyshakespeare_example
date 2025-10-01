import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Normalization: RMSNorm
# -------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.norm(dim=-1, keepdim=True) * (x.size(-1) ** -0.5)
        return x / (rms + self.eps) * self.scale

# -------------------------
# Attention + Transformer
# -------------------------
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden, n_heads, dropout=0.1):
        super().__init__()
        assert hidden % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(hidden, hidden * 3, bias=False)
        self.out = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causal_mask=None):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        att = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if causal_mask is not None:
            att = att.masked_fill(causal_mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        out = torch.matmul(att, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, hidden, ff_hidden, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, ff_hidden),
            nn.GELU(),
            nn.Linear(ff_hidden, hidden),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, hidden, n_heads, ff_hidden, dropout=0.1, norm_type="layer"):
        super().__init__()
        self.attn = MultiHeadSelfAttention(hidden, n_heads, dropout)
        self.ff = FeedForward(hidden, ff_hidden, dropout)
        if norm_type == "layer":
            self.norm1 = nn.LayerNorm(hidden)
            self.norm2 = nn.LayerNorm(hidden)
        elif norm_type == "rms":
            self.norm1 = RMSNorm(hidden)
            self.norm2 = RMSNorm(hidden)
        else:
            raise ValueError("norm_type must be 'layer' or 'rms'")
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, causal_mask=None):
        x = x + self.resid_dropout(self.attn(self.norm1(x), causal_mask=causal_mask))
        x = x + self.resid_dropout(self.ff(self.norm2(x)))
        return x

class CharTransformer(nn.Module):
    def __init__(self, vocab_size, seq_len, layers=4, hidden=256, heads=4, ff_mult=4, dropout=0.1, norm_type="layer"):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len

        self.token_emb = nn.Embedding(vocab_size, hidden)
        self.pos_emb = nn.Embedding(seq_len, hidden)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden, heads, hidden * ff_mult, dropout, norm_type)
            for _ in range(layers)
        ])
        self.final_norm = nn.LayerNorm(hidden) if norm_type == "layer" else RMSNorm(hidden)
        self.head = nn.Linear(hidden, vocab_size, bias=False)

        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
        self.register_buffer("causal_mask", mask)

    def forward(self, x):
        B, T = x.shape
        tok = self.token_emb(x)
        pos = self.pos_emb(torch.arange(T, device=x.device)).unsqueeze(0)
        h = self.drop(tok + pos)

        causal = self.causal_mask[:, :, :T, :T]
        for block in self.blocks:
            h = block(h, causal)
        h = self.final_norm(h)
        return self.head(h)

    def generate(self, start_tokens, max_new_tokens=200, temperature=1.0, sample=False, top_k=None):
        self.eval()
        device = next(self.parameters()).device
        seq = start_tokens.to(device)
        generated = seq.tolist()[0]

        for _ in range(max_new_tokens):
            context = seq[:, -self.seq_len:]
            logits = self(context)[:, -1, :] / (temperature if temperature > 0 else 1.0)
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -1e10
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1) if sample else torch.argmax(probs, dim=-1, keepdim=True)
            generated.append(int(next_token.item()))
            seq = torch.cat([seq, next_token], dim=1)

        return generated
