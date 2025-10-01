"""
Transformer model implementation for character-level text generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with causal masking."""
    
    def __init__(self, hidden_size, n_heads, dropout=0.1):
        super().__init__()
        assert hidden_size % n_heads == 0, "hidden_size must be divisible by n_heads"
        
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads
        
        # Combined QKV projection
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, T, C = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.hidden_size, dim=2)
        
        # Reshape for multi-head attention: (B, T, C) -> (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        scores = (q @ k.transpose(-2, -1)) / scale
        
        # Apply causal mask (prevent attending to future positions)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out(out)
        
        return out


class FeedForward(nn.Module):
    """Position-wise feed-forward network with GELU activation."""
    
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Single Transformer block with attention and feed-forward layers."""
    
    def __init__(self, hidden_size, n_heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = MultiHeadAttention(hidden_size, n_heads, dropout)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.ff = FeedForward(hidden_size, dropout)
    
    def forward(self, x):
        # Pre-norm architecture with residual connections
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class Transformer(nn.Module):
    """Complete Transformer model for autoregressive text generation."""
    
    def __init__(self, vocab_size, hidden_size, n_layers, n_heads, seq_length, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        
        # Token and positional embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(seq_length, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        
        # Stack of Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, n_heads, dropout)
            for _ in range(n_layers)
        ])
        
        # Final layer norm and projection to vocabulary
        self.ln_f = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights with small random values."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        """
        Forward pass through the model.
        
        Args:
            idx: Input token indices (B, T)
            targets: Target token indices for loss computation (B, T)
        
        Returns:
            logits: Output logits (B, T, vocab_size)
            loss: Cross-entropy loss (if targets provided)
        """
        B, T = idx.shape
        
        # Token embeddings + positional embeddings
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = self.dropout(tok_emb + pos_emb)
        
        # Apply Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm and projection to vocabulary
        x = self.ln_f(x)
        logits = self.head(x)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate new tokens autoregressively.
        
        Args:
            idx: Starting token indices (B, T)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top k most likely tokens
        
        Returns:
            Generated token indices (B, T + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Crop context to maximum sequence length
            idx_cond = idx if idx.size(1) <= self.seq_length else idx[:, -self.seq_length:]
            
            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Optional top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx


def count_parameters(model):
    """Count total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)