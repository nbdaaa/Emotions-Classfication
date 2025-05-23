import math
import torch
import torch.nn as nn

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super(RotaryEmbedding, self).__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, seq_len):
        # Generate a range for sequence length and reshape for broadcasting
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq).unsqueeze(1)
        # Calculate the frequency embeddings using broadcasting instead of einsum
        freqs = t * self.inv_freq.unsqueeze(0)  # Shape: [seq_len, dim//2]
        emb = torch.cat((freqs, freqs), dim=-1)  # Duplicate to match input dimension
        return emb[None, :, :]  # Shape: [1, seq_len, dim]

def apply_rotary_pos_emb(q, k, sinusoidal_pos):
    # Split the query and key tensors into even and odd dimensions
    q_cos, q_sin = q[..., 0::2], q[..., 1::2]
    k_cos, k_sin = k[..., 0::2], k[..., 1::2]

    # Split the positional encodings into cosine and sine parts
    cos, sin = sinusoidal_pos[..., 0::2], sinusoidal_pos[..., 1::2]

    # Apply rotary embeddings without einsum, element-wise operations
    q_rot = torch.cat([q_cos * cos - q_sin * sin, q_cos * sin + q_sin * cos], dim=-1)
    k_rot = torch.cat([k_cos * cos - k_sin * sin, k_cos * sin + k_sin * cos], dim=-1)

    return q_rot, k_rot

# Transformer Encoder Layer with Rotary Position Embedding
class TransformerEncoderLayerWithROPE(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayerWithROPE, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads

        self.rotary_emb = RotaryEmbedding(self.depth)

        # Linear layers for Q, K, V matrices
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        # Output linear transformation
        self.dense = nn.Linear(d_model, d_model)

        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

        # Layer normalization and dropout
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.transpose(1, 2)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))
        dk = torch.tensor(k.size(-1), dtype=torch.float32, device=q.device)
        scaled_attention_logits = matmul_qk / torch.sqrt(dk)

        if mask is not None:
            scaled_attention_logits = scaled_attention_logits.masked_fill(mask == 0, -1e9)

        attention_weights = torch.nn.functional.softmax(scaled_attention_logits, dim=-1)
        output = torch.matmul(attention_weights, v)

        return output, attention_weights

    def forward(self, x, mask=None):
        batch_size = x.size(0)
        seq_len = x.size(1)

        # Apply linear layers and split into heads
        q = self.split_heads(self.wq(x), batch_size)
        k = self.split_heads(self.wk(x), batch_size)
        v = self.split_heads(self.wv(x), batch_size)

        # Rotary position embedding
        sinusoidal_pos = self.rotary_emb(seq_len)
        q_rot, k_rot = apply_rotary_pos_emb(q, k, sinusoidal_pos)

        # Apply the custom scaled dot-product attention
        scaled_attention, _ = self.scaled_dot_product_attention(q_rot, k_rot, v, mask)

        # Transpose and reshape back to (batch_size, seq_len, d_model)
        scaled_attention = scaled_attention.transpose(1, 2).contiguous()
        concat_attention = scaled_attention.view(batch_size, -1, self.d_model)

        # Apply the final linear layer to combine the heads
        attn_output = self.dense(concat_attention)

        # Add & Norm
        x = self.layernorm1(x + self.dropout(attn_output))

        # Feed-forward
        ff_output = self.feed_forward(x)

        # Add & Norm
        x = self.layernorm2(x + self.dropout(ff_output))

        return x
    
class TransformerModelWithROPE(nn.Module):
    def __init__(self, vocab_size, embed_size, d_model, num_heads, d_ff, output_size, num_layers, dropout=0.1):
        super(TransformerModelWithROPE, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayerWithROPE(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(d_model, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.embedding(x)  # (batch_size, seq_len, embed_size)
        for layer in self.encoder_layers:
            x = layer(x, mask)  # (batch_size, seq_len, d_model)
        x = x.mean(dim=1)  # (batch_size, d_model)
        x = self.fc(self.dropout(x))  # (batch_size, output_size)
        return x