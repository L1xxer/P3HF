import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        """
        Multi-head attention module

        Args:
            d_model: Input/output dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections for Q, K, V
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: Query tensor [num_tokens, d_model]
            k: Key tensor [num_tokens, d_model]
            v: Value tensor [num_tokens, d_model]
            mask: Optional mask tensor for masked attention

        Returns:
            output: Attention output [num_tokens, d_model]
            attention: Attention weights [num_heads, num_tokens, num_tokens]
        """
        # Add batch dimension if not present
        if q.dim() == 2:
            q = q.unsqueeze(0)  # [1, num_tokens, d_model]
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)

        batch_size = q.size(0)

        # Linear projections and reshape for multi-head attention
        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Compute scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Get attention weights and apply dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention weights to values
        context = torch.matmul(attn_weights, v)

        # Reshape and concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # Final projection
        output = self.out_proj(context)

        # Remove batch dimension if input didn't have it
        if q.size(0) == 1 and q.dim() == 4:  # We added the batch dimension
            output = output.squeeze(0)
            attn_weights = attn_weights.squeeze(0)

        return output, attn_weights