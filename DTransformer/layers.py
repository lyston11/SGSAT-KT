import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

MIN_SEQ_LEN = 5


class DTransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout, kq_same=True):
        super().__init__()
        self.masked_attn_head = MultiHeadAttention(d_model, n_heads, kq_same)

        self.dropout_rate = dropout
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def device(self):
        return next(self.parameters()).device

    def forward(self, query, key, values, lens, peek_cur=False, need_scores=False):
        seqlen = query.size(1)
        mask = torch.ones(seqlen, seqlen).tril(0 if peek_cur else -1)
        mask = mask.bool()[None, None, :, :].to(self.device())

        if self.training:
            mask = mask.expand(query.size(0), -1, -1, -1).contiguous()

            for b in range(query.size(0)):
                if lens[b] < MIN_SEQ_LEN:
                    continue
                idx = random.sample(
                    range(lens[b] - 1), max(1, int(lens[b] * self.dropout_rate))
                )
                for i in idx:
                    mask[b, :, i + 1 :, i] = 0

        query_, scores = self.masked_attn_head(
            query, key, values, mask, maxout=not peek_cur, need_scores=need_scores
        )
        query = query + self.dropout(query_)
        return self.layer_norm(query), scores


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, kq_same=True, bias=True, dropout=0.2):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.h = n_heads
        self.dropout = dropout
        self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        if kq_same:
            self.k_linear = self.q_linear
        else:
            self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        self.v_linear = nn.Linear(d_model, d_model, bias=bias)

        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
        torch.nn.init.xavier_uniform_(self.gammas)

    def k_select(self, scores, k_index):
        bs, head, seqlen, _ = scores.size()
        if k_index >= seqlen:
            return F.softmax(scores, dim=-1)

        scores_a = scores[:, :, :k_index, :]
        scores_b = scores[:, :, k_index:, :].reshape(bs * head * (seqlen - k_index), -1)

        sorted_scores, _ = torch.sort(scores_b, descending=True)
        scores_t = sorted_scores[:, k_index - 1:k_index].repeat(1, seqlen)

        neg_inf = torch.finfo(scores_b.dtype).min
        sparse_scores_b = torch.where(
            scores_b - scores_t >= 0,
            scores_b,
            torch.full_like(scores_b, neg_inf),
        )
        scores_b = sparse_scores_b.reshape(bs, head, seqlen - k_index, -1)

        scores = torch.cat([scores_a, scores_b], dim=2)
        scores = F.softmax(scores, dim=-1)
        return scores

    def forward(
        self,
        q,
        k,
        v,
        mask,
        emb_type="qid",
        sparse_ratio=0.8,
        k_index=5,
        maxout=False,
        need_scores=False,
    ):
        bs = q.size(0)

        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        v_, scores = attention(
            q,
            k,
            v,
            mask,
            self.gammas,
            maxout,
            need_scores,
        )

        if need_scores and scores is not None:
            scores = self.k_select(scores, k_index)
        else:
            scores = None

        concat = v_.transpose(1, 2).contiguous().view(bs, -1, self.d_model)

        output = self.out_proj(concat)

        return output, scores


def attention(q, k, v, mask, gamma=None, maxout=False, need_scores=False):
    d_k = k.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    _, _, seqlen, _ = scores.size()

    if gamma is not None:
        x1 = torch.arange(seqlen).float().expand(seqlen, -1).to(gamma.device)
        x2 = x1.transpose(0, 1).contiguous()

        with torch.no_grad():
            neg_inf = torch.finfo(scores.dtype).min
            scores_ = scores.masked_fill(mask == 0, neg_inf)
            scores_ = F.softmax(scores_, dim=-1)

            distcum_scores = torch.cumsum(scores_, dim=-1)
            disttotal_scores = torch.sum(scores_, dim=-1, keepdim=True)
            position_effect = torch.abs(x1 - x2)[None, None, :, :]
            dist_scores = torch.clamp(
                (disttotal_scores - distcum_scores) * position_effect, min=0.0
            )
            dist_scores = dist_scores.sqrt().detach()

        gamma = -1.0 * gamma.abs().unsqueeze(0)
        total_effect = torch.clamp((dist_scores * gamma).exp(), min=1e-5, max=1e5)

        scores *= total_effect

    neg_inf = torch.finfo(scores.dtype).min
    scores.masked_fill_(mask == 0, neg_inf)
    scores = F.softmax(scores, dim=-1)
    scores = scores.masked_fill(mask == 0, 0)

    if maxout:
        scale = torch.clamp(1.0 / scores.max(dim=-1, keepdim=True)[0], max=5.0)
        scores *= scale

    output = torch.matmul(scores, v)
    if need_scores:
        return output, scores
    return output, None
