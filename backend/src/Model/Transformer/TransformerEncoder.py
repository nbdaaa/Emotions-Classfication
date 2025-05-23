import math
import torch
import torch.nn as nn
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # Create a long enough 'positional' tensor and fill it with positional encodings
        # [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (d_model/2,)

        # Khởi tạo một tensor `pe` với các giá trị bằng 0, sẽ chứa các giá trị mã hóa vị trí (positional encodings).
        # `pe` có dạng (max_len, d_model), trong đó `max_len` là chiều dài tối đa của chuỗi
        # và `d_model` là chiều kích thước của embeddings của mô hình.
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)

        # Đối với tất cả các vị trí trong chuỗi (từ 0 đến max_len-1) và đối với tất cả các chỉ số chẵn trong các chiều kích thước embedding,
        # tính toán giá trị sine của tích `position` và `div_term` và gán nó vào các vị trí tương ứng trong `pe`.
        # Gán các giá trị sine cho mọi chiều kích thước khác của embedding (bắt đầu từ chỉ số 0).
        # `position` có dạng (max_len, 1) và `div_term` có dạng (d_model/2,).
        # Phép toán `position * div_term` được broadcast thành dạng (max_len, d_model/2).
        # Kết quả được lưu trữ trong các chỉ số chẵn của `pe`, vì vậy phần được gán có dạng (max_len, d_model/2).
        pe[:, 0::2] = torch.sin(position * div_term)  # (max_len, d_model/2)

        # Tương tự, đối với tất cả các vị trí trong chuỗi và đối với tất cả các chỉ số lẻ trong các chiều kích thước embedding,
        # tính toán giá trị cosine của tích `position` và `div_term` và gán nó vào các vị trí tương ứng trong `pe`.
        # Gán các giá trị cosine cho các chiều kích thước còn lại của embedding (bắt đầu từ chỉ số 1).
        # Giống như trước, `position * div_term` được broadcast thành dạng (max_len, d_model/2).
        # Kết quả được lưu trữ trong các chỉ số lẻ của `pe`, vì vậy phần được gán có dạng (max_len, d_model/2).
        pe[:, 1::2] = torch.cos(position * div_term)  # (max_len, d_model/2)


        # Add a batch dimension and register it as a buffer
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)  # Register as buffer so it's not a parameter

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]  # Add positional encoding, (batch_size, seq_len, d_model)
        return x  # (batch_size, seq_len, d_model)
    
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads

        # Linear layers for Q, K, V matrices
        self.wq = nn.Linear(d_model, d_model)  # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        self.wk = nn.Linear(d_model, d_model)  # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        self.wv = nn.Linear(d_model, d_model)  # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)

        # Output linear transformation
        self.dense = nn.Linear(d_model, d_model)  # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)   self.

        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),  # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff)
            nn.ReLU(),
            nn.Linear(d_ff, d_model)  # (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        )

        # Layer normalization and dropout
        self.layernorm1 = nn.LayerNorm(d_model)  # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        self.layernorm2 = nn.LayerNorm(d_model)  # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x, batch_size):
        # Split the last dimension into (num_heads, depth)
        x = x.view(batch_size, -1, self.num_heads, self.depth)  # (batch_size, seq_len, d_model) -> (batch_size, seq_len, num_heads, depth)
        # Transpose the result to shape (batch_size, num_heads, seq_len, depth)
        return x.transpose(1, 2)  # (batch_size, seq_len, num_heads, depth) -> (batch_size, num_heads, seq_len, depth)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))  # (batch_size, num_heads, seq_len_q, seq_len_k)
        dk = torch.tensor(k.size(-1), dtype=torch.float32)  # scalar
        scaled_attention_logits = matmul_qk / torch.sqrt(dk)  # (batch_size, num_heads, seq_len_q, seq_len_k)

        if mask is not None:
            scaled_attention_logits = scaled_attention_logits.masked_fill(mask == 0, -1e9)

        attention_weights = torch.nn.functional.softmax(scaled_attention_logits, dim=-1)  # (batch_size, num_heads, seq_len_q, seq_len_k)
        output = torch.matmul(attention_weights, v)  # (batch_size, num_heads, seq_len_q, depth_v)

        return output, attention_weights  # (batch_size, num_heads, seq_len_q, depth_v), (batch_size, num_heads, seq_len_q, seq_len_k)

    def forward(self, x, mask=None):
        batch_size = x.size(0)  # (batch_size, seq_len, d_model)

        # Apply linear layers and split into heads
        q = self.split_heads(self.wq(x), batch_size)  # (batch_size, num_heads, seq_len, depth)
        k = self.split_heads(self.wk(x), batch_size)  # (batch_size, num_heads, seq_len, depth)
        v = self.split_heads(self.wv(x), batch_size)  # (batch_size, num_heads, seq_len, depth)

        # Apply the custom scaled dot-product attention
        scaled_attention, _ = self.scaled_dot_product_attention(q, k, v, mask)  # (batch_size, num_heads, seq_len_q, depth_v)

        # Transpose and reshape back to (batch_size, seq_len, d_model)
        scaled_attention = scaled_attention.transpose(1, 2).contiguous()  # (batch_size, seq_len, num_heads, depth)
        concat_attention = scaled_attention.view(batch_size, -1, self.d_model)  # (batch_size, seq_len, d_model)

        # Apply the final linear layer to combine the heads
        attn_output = self.dense(concat_attention)  # (batch_size, seq_len, d_model)

        # Add & Norm
        x = self.layernorm1(x + self.dropout(attn_output))  # (batch_size, seq_len, d_model)

        # Feed-forward
        ff_output = self.feed_forward(x)  # (batch_size, seq_len, d_model)

        # Add & Norm
        x = self.layernorm2(x + self.dropout(ff_output))  # (batch_size, seq_len, d_model)

        return x  # (batch_size, seq_len, d_model)

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size, d_model, num_heads, d_ff, output_size, num_layers, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = PositionalEncoding(d_model)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(d_model, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.embedding(x)  # (batch_size, seq_len, embed_size)
        x = self.positional_encoding(x)  # (batch_size, seq_len, d_model)
        for layer in self.encoder_layers:
            x = layer(x, mask)  # (batch_size, seq_len, d_model)
        x = x.mean(dim=1)  # (batch_size, d_model)
        x = self.fc(self.dropout(x))  # (batch_size, output_size)
        return x
