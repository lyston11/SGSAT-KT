"""
预计算嵌入加载与融合层（v4: Cross-Attention 融合 + 辅助 InfoNCE 对比损失）
ID embedding 作为 Query 主动 attend LLM 语义特征，提取 task-specific 信号，
通过门控残差防止 attention 坍塌，并保留辅助对比损失。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from DTransformer.precomputed import PrecomputedEmbeddings


class PrecomputedEmbeddingLayer(nn.Module):
    """
    预计算嵌入层（v2: 多层漏斗式投影）

    将 LLM 嵌入通过  hidden_size → inter_dim → proj_dim 的多层投影，
    保留更多语义信息，不再使用单层 Linear 直接压到 d_model。
    """
    def __init__(self, precomputed_embeddings, llm_proj_dim=256, llm_inter_dim=512, use_llm=True):
        super().__init__()
        self.use_llm = use_llm
        self.llm_proj_dim = llm_proj_dim
        self.proj_q = None
        self.proj_kc = None
        self.W_p = None

        if use_llm and precomputed_embeddings is not None:
            hidden_size = precomputed_embeddings.hidden_size

            self.proj_q = nn.Sequential(
                nn.Linear(hidden_size, llm_inter_dim),
                nn.GELU(),
                nn.LayerNorm(llm_inter_dim),
                nn.Dropout(0.1),
                nn.Linear(llm_inter_dim, llm_proj_dim),
                nn.LayerNorm(llm_proj_dim),
            )
            self.proj_kc = nn.Sequential(
                nn.Linear(hidden_size, llm_inter_dim),
                nn.GELU(),
                nn.LayerNorm(llm_inter_dim),
                nn.Dropout(0.1),
                nn.Linear(llm_inter_dim, llm_proj_dim),
                nn.LayerNorm(llm_proj_dim),
            )
            self.W_p = nn.Linear(llm_proj_dim, llm_proj_dim, bias=False)
            self.precomputed = precomputed_embeddings
            print(f"✅ 使用预计算嵌入层(v2多层投影): {hidden_size} -> {llm_inter_dim} -> {llm_proj_dim}")
        else:
            self.precomputed = None
            print("⚠️  未使用预计算嵌入")

    def _get_dtype(self):
        if self.proj_q is not None:
            for module in self.proj_q:
                if isinstance(module, nn.Linear):
                    return module.weight.dtype
        return torch.float32

    def forward(self, q_ids, kc_ids=None):
        if not self.use_llm or self.precomputed is None:
            batch_size, seq_len = q_ids.shape
            return torch.zeros(
                batch_size, seq_len, self.llm_proj_dim,
                device=q_ids.device, dtype=self._get_dtype(),
            )

        batch_size, seq_len = q_ids.shape
        q_ids_flat = q_ids.cpu().numpy().flatten().tolist()

        q_embs_np = self.precomputed.get_batch_question_embeddings(q_ids_flat)
        q_embs_np = q_embs_np.reshape(batch_size, seq_len, -1)

        q_embs = torch.from_numpy(q_embs_np).to(
            device=q_ids.device,
            dtype=self._get_dtype(),
        )

        e_q = self.proj_q(q_embs)

        if kc_ids is not None:
            kc_ids_flat = kc_ids.cpu().numpy().flatten().tolist()
            kc_embs_np = self.precomputed.get_batch_kc_embeddings(kc_ids_flat)
            kc_embs_np = kc_embs_np.reshape(batch_size, seq_len, -1)
            kc_embs = torch.from_numpy(kc_embs_np).to(
                device=kc_ids.device,
                dtype=self._get_dtype(),
            )

            e_kc = self.proj_kc(kc_embs)
            e_q = e_q + self.W_p(e_kc)

        return e_q


class LLMGroundingWithPrecomputed(nn.Module):
    """
    结合ID嵌入和预计算LLM语义嵌入的混合嵌入模块（v4: Cross-Attention 融合）
    """
    def __init__(
        self,
        n_questions,
        d_model=256,
        id_dim=128,
        llm_proj_dim=256,
        llm_inter_dim=512,
        precomputed_embeddings=None,
        use_llm=True,
        id_dropout_rate=0.15,
        num_heads=4,
        dropout=0.1,
    ):
        super().__init__()
        self.use_llm = use_llm
        self.d_model = d_model
        self.id_dim = id_dim
        self.llm_proj_dim = llm_proj_dim

        self.q_embed = nn.Embedding(n_questions + 1, id_dim)

        if use_llm and precomputed_embeddings is not None:
            self.llm_layer = PrecomputedEmbeddingLayer(
                precomputed_embeddings,
                llm_proj_dim=llm_proj_dim,
                llm_inter_dim=llm_inter_dim,
                use_llm=True,
            )
            self.id_to_llm_proj = nn.Linear(id_dim, llm_proj_dim)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=llm_proj_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.gate = nn.Sequential(
                nn.Linear(llm_proj_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Sigmoid(),
            )
            self.id_dropout = nn.Dropout(p=id_dropout_rate)
            self.fusion_norm = nn.LayerNorm(llm_proj_dim)
            self.contrast_head = nn.Linear(llm_proj_dim, 128)

            print(f"✅ Cross-Attention融合+对比损失(v4): id_dim={id_dim}, llm_proj_dim={llm_proj_dim}, "
                  f"num_heads={num_heads}, id_dropout={id_dropout_rate}")
        else:
            self.llm_layer = None
            self.id_proj = nn.Linear(id_dim, d_model)

    def forward(self, q_ids, kc_ids=None):
        id_emb = self.q_embed(q_ids)

        if not self.use_llm or self.llm_layer is None:
            return self.id_proj(id_emb)

        if self.training:
            id_emb = self.id_dropout(id_emb)

        llm_emb = self.llm_layer(q_ids, kc_ids)
        id_proj = self.id_to_llm_proj(id_emb)

        seq_len = q_ids.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=q_ids.device), diagonal=1
        ).bool()

        attn_output, _ = self.cross_attn(
            query=id_proj,
            key=llm_emb,
            value=llm_emb,
            attn_mask=causal_mask,
            need_weights=False,
        )

        gate_val = self.gate(attn_output)
        fused = gate_val * attn_output + (1 - gate_val) * id_proj

        return self.fusion_norm(fused)

    def compute_contrastive_loss(self, q_ids, kc_ids=None, temperature=0.07):
        if self.llm_layer is None:
            return torch.tensor(0.0, device=q_ids.device, requires_grad=True)

        with torch.amp.autocast(device_type="cuda", enabled=False):
            llm_emb = self.llm_layer(q_ids, kc_ids)
        _, _, dim = llm_emb.shape

        llm_flat = llm_emb.reshape(-1, dim)
        z = self.contrast_head(llm_flat)
        z = F.normalize(z, p=2, dim=-1)

        n_tokens = z.size(0)
        max_tokens = 512
        if n_tokens > max_tokens:
            indices = torch.randperm(n_tokens, device=z.device)[:max_tokens]
            z = z[indices]
            n_tokens = max_tokens

        sim_matrix = torch.matmul(z, z.T) / temperature

        if kc_ids is not None:
            kc_flat = kc_ids.reshape(-1)
            if kc_flat.size(0) > max_tokens:
                kc_flat = kc_flat[indices]
            kc_mask = kc_flat.unsqueeze(0) == kc_flat.unsqueeze(1)
        else:
            kc_mask = torch.eye(n_tokens, device=z.device, dtype=torch.bool)

        self_mask = torch.eye(n_tokens, device=z.device, dtype=torch.bool)
        positive_mask = kc_mask & ~self_mask

        sim_max = sim_matrix.detach().max(dim=-1, keepdim=True).values
        logits = sim_matrix - sim_max

        exp_logits = torch.exp(logits) * ~self_mask
        log_sum_exp = torch.log(exp_logits.sum(dim=-1, keepdim=True) + 1e-8)

        has_pos = positive_mask.sum(dim=-1) > 0
        if has_pos.sum() == 0:
            return torch.tensor(0.0, device=q_ids.device, requires_grad=True)

        pos_logits = (logits * positive_mask).sum(dim=-1)
        n_pos = positive_mask.sum(dim=-1).clamp(min=1)
        pos_mean_logit = pos_logits / n_pos

        loss_per_sample = -pos_mean_logit + log_sum_exp.squeeze(-1)
        loss = loss_per_sample[has_pos].mean()

        return loss
