"""
预计算嵌入加载器（v4: Cross-Attention 融合 + 辅助 InfoNCE 对比损失）
ID embedding 作为 Query 主动 attend LLM 语义特征，提取 task-specific 信号，
通过门控残差防止 attention 坍塌，并保留辅助对比损失。
"""
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PrecomputedEmbeddings:
    """预计算嵌入加载器"""

    def __init__(self, embedding_dir="data/embeddings"):
        self.embedding_dir = embedding_dir
        self.question_embeddings = None
        self.kc_embeddings = None
        self.question_ids = None
        self.kc_ids = None
        self.question_id_to_idx = None
        self.kc_id_to_idx = None
        self.hidden_size = None

    def load_question_embeddings(self, path=None):
        """加载题目嵌入"""
        if path is None:
            path = os.path.join(self.embedding_dir, "question_embeddings.pkl")

        if not os.path.exists(path):
            raise FileNotFoundError(f"题目嵌入文件不存在: {path}")

        print(f"📦 加载题目嵌入: {path}")
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.question_embeddings = data["embeddings"]
        self.question_ids = data["question_ids"]
        self.question_id_to_idx = {int(qid): idx for idx, qid in enumerate(self.question_ids)}
        self.hidden_size = data.get("hidden_size", 2560)

        print(f"✅ 加载了 {len(self.question_ids)} 个题目嵌入")
        print(f"📊 嵌入维度: {self.hidden_size}")

        return self.question_embeddings

    def load_kc_embeddings(self, path=None):
        """加载知识点嵌入"""
        if path is None:
            path = os.path.join(self.embedding_dir, "kc_embeddings.pkl")

        if not os.path.exists(path):
            raise FileNotFoundError(f"知识点嵌入文件不存在: {path}")

        print(f"📦 加载知识点嵌入: {path}")
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.kc_embeddings = data["embeddings"]
        self.kc_ids = data["kc_ids"]
        self.kc_id_to_idx = {int(kid): idx for idx, kid in enumerate(self.kc_ids)}
        if self.hidden_size is None:
            self.hidden_size = data.get("hidden_size", 2560)

        print(f"✅ 加载了 {len(self.kc_ids)} 个知识点嵌入")
        print(f"📊 嵌入维度: {self.hidden_size}")

        return self.kc_embeddings

    def get_question_embedding(self, question_id):
        """获取单个题目的嵌入"""
        if self.question_embeddings is None:
            raise ValueError("题目嵌入未加载，请先调用 load_question_embeddings()")

        idx = self.question_id_to_idx.get(int(question_id)) if self.question_id_to_idx else None
        if idx is None:
            return np.zeros(self.hidden_size, dtype=np.float32)

        return np.asarray(self.question_embeddings[idx], dtype=np.float32)

    def get_kc_embedding(self, kc_id):
        """获取单个知识点的嵌入"""
        if self.kc_embeddings is None:
            raise ValueError("知识点嵌入未加载，请先调用 load_kc_embeddings()")

        idx = self.kc_id_to_idx.get(int(kc_id)) if self.kc_id_to_idx else None
        if idx is None:
            return np.zeros(self.hidden_size, dtype=np.float32)

        return np.asarray(self.kc_embeddings[idx], dtype=np.float32)

    def get_batch_question_embeddings(self, question_ids):
        """批量获取题目嵌入"""
        return np.stack(
            [self.get_question_embedding(qid) for qid in question_ids],
            axis=0,
        ).astype(np.float32)

    def get_batch_kc_embeddings(self, kc_ids):
        """批量获取知识点嵌入"""
        return np.stack(
            [self.get_kc_embedding(kid) for kid in kc_ids],
            axis=0,
        ).astype(np.float32)


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

            # 多层漏斗式投影: hidden_size → llm_inter_dim → llm_proj_dim
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
        """获取投影层的数据类型"""
        if self.proj_q is not None:
            # Sequential 中第一个是 Linear，直接取其 weight.dtype
            for module in self.proj_q:
                if isinstance(module, nn.Linear):
                    return module.weight.dtype
        return torch.float32

    def forward(self, q_ids, kc_ids=None):
        """
        Args:
            q_ids: 题目ID (batch, seq_len)
            kc_ids: 知识点ID (batch, seq_len) [可选]
        Returns:
            q_emb: 题目嵌入 (batch, seq_len, llm_proj_dim)
        """
        if not self.use_llm or self.precomputed is None:
            batch_size, seq_len = q_ids.shape
            return torch.zeros(
                batch_size, seq_len, self.llm_proj_dim,
                device=q_ids.device, dtype=self._get_dtype(),
            )

        batch_size, seq_len = q_ids.shape
        q_ids_flat = q_ids.cpu().numpy().flatten().tolist()

        # 获取预计算嵌入
        q_embs_np = self.precomputed.get_batch_question_embeddings(q_ids_flat)
        q_embs_np = q_embs_np.reshape(batch_size, seq_len, -1)

        # 转换为张量
        q_embs = torch.from_numpy(q_embs_np).to(
            device=q_ids.device,
            dtype=self._get_dtype(),
        )

        # 多层投影
        e_q = self.proj_q(q_embs)

        # 如果有知识点，融合知识点嵌入
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

    v4.0 改动:
    - Cross-Attention: ID 作为 Query 主动 attend LLM 特征，提取 task-specific 语义
    - 门控残差: 2层 gate network 输出标量，防止 attention 坍塌
    - ID Dropout: 训练时以 p=0.15 将 ID 嵌入置零，迫使模型依赖 LLM
    - 辅助 InfoNCE: 在 LLM 投影空间施加对比损失

    维度流:
      id_emb [B,L,128] → id_to_llm_proj → [B,L,256]  (Query)
      llm_emb [B,L,256]                                (Key, Value)
      cross_attn(Q=id_proj, K=llm, V=llm) → attn_out [B,L,256]
      gate_val = gate_net(attn_out) → [B,L,1]
      fused = gate * attn_out + (1-gate) * id_proj → [B,L,256]
      output = LayerNorm(fused) → [B,L,256] = d_model
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

        # ID 嵌入（固定 id_dim 维，不跟随 d_model）
        self.q_embed = nn.Embedding(n_questions + 1, id_dim)

        if use_llm and precomputed_embeddings is not None:
            # 预计算 LLM 语义嵌入（多层投影到 llm_proj_dim）
            self.llm_layer = PrecomputedEmbeddingLayer(
                precomputed_embeddings,
                llm_proj_dim=llm_proj_dim,
                llm_inter_dim=llm_inter_dim,
                use_llm=True,
            )

            # ID → LLM 维度投影（128 → 256），用于 Query
            self.id_to_llm_proj = nn.Linear(id_dim, llm_proj_dim)

            # v4: Cross-Attention（ID Query → attend LLM Key/Value）
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=llm_proj_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )

            # v4: 门控网络（2层→标量，防止 attention 坍塌）
            self.gate = nn.Sequential(
                nn.Linear(llm_proj_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Sigmoid(),
            )

            # ID Dropout
            self.id_dropout = nn.Dropout(p=id_dropout_rate)

            # 融合后 LayerNorm（输出 llm_proj_dim = d_model）
            self.fusion_norm = nn.LayerNorm(llm_proj_dim)

            # 对比损失投影头（降维到 128 加速计算）
            self.contrast_head = nn.Linear(llm_proj_dim, 128)

            print(f"✅ Cross-Attention融合+对比损失(v4): id_dim={id_dim}, llm_proj_dim={llm_proj_dim}, "
                  f"num_heads={num_heads}, id_dropout={id_dropout_rate}")
        else:
            self.llm_layer = None
            # 无 LLM 时直接投影 ID 到 d_model
            self.id_proj = nn.Linear(id_dim, d_model)

    def forward(self, q_ids, kc_ids=None):
        """
        Args:
            q_ids: 题目ID (batch, seq_len)
            kc_ids: 知识点ID (batch, seq_len) [可选]
        Returns:
            q_emb: 混合嵌入 (batch, seq_len, d_model)
        """
        id_emb = self.q_embed(q_ids)  # (batch, seq_len, id_dim)

        if not self.use_llm or self.llm_layer is None:
            return self.id_proj(id_emb)

        # ID Dropout（仅训练时）
        if self.training:
            id_emb = self.id_dropout(id_emb)

        # 获取预计算的 LLM 嵌入 (batch, seq_len, llm_proj_dim)
        llm_emb = self.llm_layer(q_ids, kc_ids)

        # ID 投影到 LLM 维度 → 作为 Query
        id_proj = self.id_to_llm_proj(id_emb)  # [B, L, llm_proj_dim]

        # Causal mask（KT 不能看未来位置）
        seq_len = q_ids.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=q_ids.device), diagonal=1
        ).bool()

        # Cross-Attention: ID Query → attend LLM Key/Value
        attn_output, _ = self.cross_attn(
            query=id_proj,
            key=llm_emb,
            value=llm_emb,
            attn_mask=causal_mask,
            need_weights=False,
        )  # [B, L, llm_proj_dim]

        # 门控残差融合
        gate_val = self.gate(attn_output)  # [B, L, 1]
        fused = gate_val * attn_output + (1 - gate_val) * id_proj  # [B, L, llm_proj_dim]

        return self.fusion_norm(fused)  # [B, L, llm_proj_dim] = d_model

    def compute_contrastive_loss(self, q_ids, kc_ids=None, temperature=0.07):
        """
        v3: 辅助 InfoNCE 对比损失

        在 LLM 投影空间上施加 in-batch 对比损失，强制同 KC 题目的
        LLM 投影靠近，不同 KC 题目的投影远离。

        Args:
            q_ids: 题目ID (batch, seq_len)
            kc_ids: 知识点ID (batch, seq_len) [可选，用于正样本配对]
            temperature: InfoNCE 温度
        Returns:
            loss: scalar
        """
        if self.llm_layer is None:
            return torch.tensor(0.0, device=q_ids.device, requires_grad=True)

        # 获取 LLM 投影特征（不经过 gate，直接取投影输出）
        with torch.amp.autocast(device_type='cuda', enabled=False):
            llm_emb = self.llm_layer(q_ids, kc_ids)  # [B, L, llm_proj_dim]
        batch_size, seq_len, dim = llm_emb.shape

        # Flatten: [B*L, llm_proj_dim]
        llm_flat = llm_emb.reshape(-1, dim)

        # 对比头降维 + L2 normalize
        z = self.contrast_head(llm_flat)     # [B*L, 128]
        z = F.normalize(z, p=2, dim=-1)

        # 限制采样数量以控制显存（取前 512 个 token）
        n_tokens = z.size(0)
        max_tokens = 512
        if n_tokens > max_tokens:
            indices = torch.randperm(n_tokens, device=z.device)[:max_tokens]
            z = z[indices]
            n_tokens = max_tokens

        # 相似度矩阵
        sim_matrix = torch.matmul(z, z.T) / temperature  # [N, N]

        # 构造正样本掩码
        if kc_ids is not None:
            kc_flat = kc_ids.reshape(-1)  # [B*L]
            if kc_flat.size(0) > max_tokens:
                kc_flat = kc_flat[indices]
            # 同一 KC 的 token 对为正样本
            kc_mask = kc_flat.unsqueeze(0) == kc_flat.unsqueeze(1)  # [N, N]
        else:
            # 无 KC 信息: 每个 token 自身为正样本（标准 in-batch）
            kc_mask = torch.eye(n_tokens, device=z.device, dtype=torch.bool)

        # 排除自身（对角线）
        self_mask = torch.eye(n_tokens, device=z.device, dtype=torch.bool)
        positive_mask = kc_mask & ~self_mask  # [N, N]

        # InfoNCE loss
        # 对数值稳定性的处理：减去最大值
        sim_max = sim_matrix.detach().max(dim=-1, keepdim=True).values
        logits = sim_matrix - sim_max

        # 分母：所有非自身样本
        exp_logits = torch.exp(logits) * ~self_mask
        log_sum_exp = torch.log(exp_logits.sum(dim=-1, keepdim=True) + 1e-8)

        # 分子：正样本
        # 只计算有正样本的行
        has_pos = positive_mask.sum(dim=-1) > 0
        if has_pos.sum() == 0:
            # 没有正样本对，返回零损失
            return torch.tensor(0.0, device=q_ids.device, requires_grad=True)

        pos_logits = (logits * positive_mask).sum(dim=-1)  # 对正样本求和
        n_pos = positive_mask.sum(dim=-1).clamp(min=1)     # 正样本数量
        pos_mean_logit = pos_logits / n_pos

        # loss = -log(exp(pos) / sum_exp(all))
        loss_per_sample = -pos_mean_logit + log_sum_exp.squeeze(-1)
        loss = loss_per_sample[has_pos].mean()

        return loss
