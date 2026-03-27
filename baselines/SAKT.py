# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import math
# import copy
#
#
# def future_mask(seq_len, device):
#     """Generate a future mask for self-attention."""
#     mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
#     return mask.unsqueeze(0).unsqueeze(0).to(device)
#
#
# def clone_module(module, num):
#     """Clone a module multiple times for stackable layers."""
#     return nn.ModuleList([copy.deepcopy(module) for _ in range(num)])
#
#
# def attention(query, key, value, mask=None, dropout=None):
#     """Scaled dot-product attention."""
#     scores = torch.matmul(query, key.transpose(-2, -1))
#     scores = scores / math.sqrt(query.size(-1))  # Scale by sqrt(d_k)
#     if mask is not None:
#         scores = scores.masked_fill(mask, -1e9)
#     prob_attn = F.softmax(scores, dim=-1)
#     if dropout:
#         prob_attn = dropout(prob_attn)
#     return torch.matmul(prob_attn, value), prob_attn
#
#
# def relative_attention(query, key, value, pos_key_embeds, pos_value_embeds, mask=None, dropout=None):
#     """Relative position attention with embeddings."""
#     assert pos_key_embeds.num_embeddings == pos_value_embeds.num_embeddings
#
#     scores = torch.matmul(query, key.transpose(-2, -1))
#
#     # Compute relative position indices
#     idxs = torch.arange(scores.size(-1), device=query.device).view(-1, 1) - torch.arange(scores.size(-1), device=query.device).view(1, -1)
#     idxs = torch.clamp(idxs, 0, pos_key_embeds.num_embeddings - 1)
#
#     # Position key embeddings
#     pos_key = pos_key_embeds(idxs).transpose(-2, -1)
#     pos_scores = torch.matmul(query.unsqueeze(-2), pos_key)
#     scores = scores.unsqueeze(-2) + pos_scores
#     scores = scores / math.sqrt(query.size(-1))
#
#     # Position value embeddings
#     pos_value = pos_value_embeds(idxs)
#     value = value.unsqueeze(-3) + pos_value
#
#     if mask is not None:
#         scores = scores.masked_fill(mask.unsqueeze(-2), -1e9)
#     prob_attn = F.softmax(scores, dim=-1)
#     if dropout:
#         prob_attn = dropout(prob_attn)
#
#     output = torch.matmul(prob_attn, value).squeeze(-2)
#     return output, prob_attn.squeeze(-2)
#
#
# class MultiHeadedAttention(nn.Module):
#     def __init__(self, embed_size, num_heads, dropout_prob):
#         super().__init__()
#         assert embed_size % num_heads == 0
#         self.embed_size = embed_size
#         self.num_heads = num_heads
#         self.head_size = embed_size // num_heads
#         self.linear_layers = clone_module(nn.Linear(embed_size, embed_size), 3)
#         self.dropout = nn.Dropout(p=dropout_prob)
#
#     def forward(self, query, key, value, encode_pos, pos_key_embeds, pos_value_embeds, mask=None):
#         batch_size, seq_len = query.size(0), query.size(1)
#
#         # Apply mask for all heads
#         if mask is not None:
#             mask = mask.unsqueeze(1)
#
#         # Linear projections and split into heads
#         query, key, value = [l(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
#                              for l, x in zip(self.linear_layers, (query, key, value))]
#
#         if encode_pos:
#             out, _ = relative_attention(query, key, value, pos_key_embeds, pos_value_embeds, mask, self.dropout)
#         else:
#             out, _ = attention(query, key, value, mask, self.dropout)
#
#         # Combine heads
#         out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_size)
#         return out
#
#
# class SAKT(nn.Module):
#     def __init__(self, ex_total, dim, heads, num_layers, encode_pos, max_pos, dropout, device='cpu'):
#         super(SAKT, self).__init__()
#         self.device = device
#         self.dim = dim
#         self.encode_pos = encode_pos
#
#         # Embedding layers
#         self.item_embeds = nn.Embedding(ex_total + 1, dim // 2, padding_idx=0)
#         self.skill_embeds = nn.Embedding(ex_total + 1, dim // 2, padding_idx=0)
#         self.pos_key_embeds = nn.Embedding(max_pos, dim // heads)
#         self.pos_value_embeds = nn.Embedding(max_pos, dim // heads)
#
#         # Input and attention layers
#         self.input_proj = nn.Linear(2 * dim, dim)
#         self.attn_layers = nn.ModuleList([MultiHeadedAttention(dim, heads, dropout) for _ in range(num_layers)])
#         self.dropout = nn.Dropout(p=dropout)
#         self.output_proj = nn.Linear(dim, 1)
#
#     def forward(self, q, s):
#         """
#         Forward method to process q and s.
#         q: Exercise IDs or queries.
#         s: Labels indicating correctness.
#         """
#         batch_size, seq_len = q.size(0), q.size(1)
#
#         # Check and clamp indices to valid range
#         q = torch.clamp(q, min=0, max=self.item_embeds.num_embeddings - 1)
#
#         # Prepare embeddings
#         item_emb = self.item_embeds(q)
#         skill_emb = self.skill_embeds(q)
#         label_inputs = s.unsqueeze(-1).float()
#         inputs = torch.cat([item_emb, skill_emb, item_emb, skill_emb], dim=-1)
#         inputs[..., :self.dim] *= label_inputs
#         inputs[..., self.dim:] *= 1 - label_inputs
#         inputs = F.relu(self.input_proj(inputs))
#
#         # Generate query
#         query = torch.cat([self.item_embeds(q), self.skill_embeds(q)], dim=-1)
#
#         # Generate mask
#         mask = future_mask(seq_len, self.device)
#         if inputs.is_cuda:
#             mask = mask.cuda()
#
#         # Process attention layers
#         outputs = inputs
#         for attn_layer in self.attn_layers:
#             outputs = attn_layer(query, outputs, outputs, mask)
#
#         # Output projection
#         outputs = self.output_proj(outputs)
#         return outputs.squeeze(-1)
#
#     def get_loss(self, q, s, pid=None):
#         """
#         Loss computation method.
#         q: Exercise IDs or queries.
#         s: Labels indicating correctness.
#         pid: Optional parameter (not used in this implementation).
#         """
#         # Clamp indices to valid range
#         q = torch.clamp(q, min=0, max=self.item_embeds.num_embeddings - 1)
#         s = torch.clamp(s, min=0, max=1)  # Ensure labels are valid binary values
#
#         logits = self.forward(q, s)
#
#         # Mask to exclude padding positions
#         valid_mask = s >= 0
#         masked_logits = logits[valid_mask]
#         masked_targets = s[valid_mask].float()
#
#         # Binary Cross-Entropy Loss
#         loss = F.binary_cross_entropy_with_logits(masked_logits, masked_targets)
#         return loss
#
#     def predict(self, q, s, pid=None):
#         """
#         Predict method for compatibility with train.py.
#         q: Exercise IDs or queries.
#         s: Labels indicating correctness.
#         pid: Optional parameter (not used in this implementation).
#         """
#         # Forward pass to get logits
#         logits = self.forward(q, s)
#
#         # Apply sigmoid to convert logits to probabilities
#         probabilities = torch.sigmoid(logits)
#
#         # For compatibility with train.py, return probabilities
#         # The second return value (e.g., hidden states) is not applicable here
#         return probabilities, None, None, 0.0, None



import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SAKT(nn.Module):
    def __init__(self, n_questions, d_model=128, n_heads=8, n_layers=1, dropout=0.1, device="cpu"):
        super(SAKT, self).__init__()

        self.device = device
        self.n_questions = n_questions
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers

        self.q_embed = nn.Embedding(n_questions + 1, d_model)
        self.s_embed = nn.Embedding(2, d_model)

        self.blocks = nn.ModuleList([SAKTLayers(d_model, n_heads, dropout) for _ in range(n_layers)])

        self.out = nn.Sequential(
            nn.Linear(2 * d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        self.to(self.device)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)

    def forward(self, q_emb, s_emb, mask=None):
        seq_len = q_emb.size(1)
        if mask is None:
            mask = future_mask(seq_len, self.device)

        # Apply transformer blocks
        for block in self.blocks:
            q_emb, s_emb = block(q_emb, s_emb, mask)

        return q_emb

    def predict(self, q, s, pid=None):
        # Preprocessing inputs
        q = q.masked_fill(q < 0, 0)
        s = s.masked_fill(s < 0, 0)

        q_emb = self.q_embed(q)
        s_emb = self.s_embed(s) + q_emb

        seq_len = q.size(1)
        mask = future_mask(seq_len, self.device)

        h = self.forward(q_emb, s_emb, mask)

        y = self.out(torch.cat([q_emb, h], dim=-1)).squeeze(-1)
        return y

    def get_loss(self, q, s, pid=None):
        # Compute predictions
        logits = self.predict(q, s, pid)

        # Flatten logits and labels to compute BCE loss
        masked_labels = s[s >= 0].float()
        masked_logits = logits[s >= 0]

        # Use BCE loss
        loss = F.binary_cross_entropy_with_logits(masked_logits, masked_labels, reduction="mean")
        return loss


class SAKTLayers(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()

        # Multi-head attention
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, q_emb, s_emb, mask):
        # Multi-head attention
        attn_output = self.attn(q_emb, q_emb, s_emb, mask)
        q_emb = self.layer_norm1(q_emb + self.dropout1(attn_output))

        # Feed-forward network
        ffn_output = self.ffn(q_emb)
        q_emb = self.layer_norm2(q_emb + self.dropout2(ffn_output))

        return q_emb, s_emb


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, kq_same=True, bias=True):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.h = n_heads

        # Linear projections
        self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        if kq_same:
            self.k_linear = self.q_linear
        else:
            self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        self.v_linear = nn.Linear(d_model, d_model, bias=bias)

        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, q, k, v, mask):
        bs = q.size(0)

        # Linear projections and split into heads
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(bs, -1, self.d_model)

        # Output projection
        return self.out_proj(attn_output)


def future_mask(seq_len, device):
    """Generate a mask for future tokens."""
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return mask.unsqueeze(0).unsqueeze(0).to(device)

