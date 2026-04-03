"""TriSG-KT: Triple-Sparse Semantic Graph Knowledge Tracing."""
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from DTransformer.embedding_loader import LLMGroundingWithPrecomputed
from DTransformer.graph import DCFSimGraphEnhanced, GNNPrerequisiteGraph
from DTransformer.grounding import LLMGrounding, LLMGroundingWithID
from DTransformer.layers import DTransformerLayer, MultiHeadAttention, attention

MIN_SEQ_LEN = 5


class DTransformer(nn.Module):
    """
    TriSG-KT 主模型
    
    三类关键增强:
    - 语义增强: LLM 语义 grounding 与 SSA 对齐
    - 结构增强: GNN prerequisite graph
    - 优化/分析增强: graph-enhanced similarity 与对比学习
    """
    def __init__(
        self,
        n_questions,
        n_pid=0,
        d_model=256,
        d_fc=512,
        n_heads=8,
        n_know=16,
        n_layers=1,
        dropout=0.05,
        lambda_cl=0.1,
        proj=False,
        hard_neg=True,
        window=1,
        shortcut=False,
        # ============ 修改点1参数: LLM语义Grounding ============
        use_llm=False,
        pretrained_model="pretrained_models/qwen3-4b",
        freeze_bert=True,
        precomputed_embeddings=None,
        id_dim=128,
        llm_proj_dim=256,
        llm_inter_dim=512,
        id_dropout_rate=0.15,
        lambda_contra=0.3,
        contrast_temperature=0.07,
        cross_attn_heads=4,
        # ============ 修改点2参数: GNN先决图 ============
        n_kc=100,
        use_gnn=False,
        gnn_layers=2,
    ):
        super().__init__()
        self.n_questions = n_questions
        self.n_kc = n_kc
        self.use_gnn = use_gnn
        self.use_llm = use_llm
        self.use_precomputed_llm = False
        self.id_dim = id_dim
        self.llm_proj_dim = llm_proj_dim
        self.llm_inter_dim = llm_inter_dim
        self.lambda_contra = lambda_contra
        self.contrast_temperature = contrast_temperature

        # ============ 修改点1: LLM语义Grounding嵌入层 ============
        if use_llm:
            if precomputed_embeddings is not None:
                self.q_embed = LLMGroundingWithPrecomputed(
                    n_questions,
                    d_model=d_model,
                    id_dim=id_dim,
                    llm_proj_dim=llm_proj_dim,
                    llm_inter_dim=llm_inter_dim,
                    precomputed_embeddings=precomputed_embeddings,
                    use_llm=True,
                    id_dropout_rate=id_dropout_rate,
                    num_heads=cross_attn_heads,
                    dropout=dropout,
                )
                self.use_precomputed_llm = True
            else:
                self.q_embed = LLMGroundingWithID(
                    n_questions,
                    d_model=d_model,
                    id_dim=id_dim,
                    llm_proj_dim=llm_proj_dim,
                    llm_inter_dim=llm_inter_dim,
                    pretrained_model=pretrained_model,
                    freeze_bert=freeze_bert,
                    use_llm=True,
                    id_dropout_rate=id_dropout_rate,
                )
        else:
            # 纯 ID 嵌入层（向后兼容）
            self.q_embed = nn.Embedding(n_questions + 1, d_model)
        
        self.s_embed = nn.Embedding(2, d_model)

        if n_pid > 0:
            self.q_diff_embed = nn.Embedding(n_questions + 1, d_model)
            self.s_diff_embed = nn.Embedding(2, d_model)
            self.p_diff_embed = nn.Embedding(n_pid + 1, 1)

        # ============ 修改点2: GNN先决图嵌入 ============
        if use_gnn:
            self.gnn = GNNPrerequisiteGraph(n_kc, d_model, gnn_layers, dropout)

        # Embedding 层 dropout（正则化）
        self.emb_dropout = nn.Dropout(dropout)

        # 遗忘机制：重复次数嵌入
        self.max_repeats = 20
        self.repeat_embed = nn.Embedding(self.max_repeats, d_model)

        self.n_heads = n_heads
        self.block1 = DTransformerLayer(d_model, n_heads, dropout)
        self.block2 = DTransformerLayer(d_model, n_heads, dropout)
        self.block3 = DTransformerLayer(d_model, n_heads, dropout)
        self.block4 = DTransformerLayer(d_model, n_heads, dropout, kq_same=False)
        self.block5 = DTransformerLayer(d_model, n_heads, dropout)

        self.n_know = n_know
        self.know_params = nn.Parameter(torch.empty(n_know, d_model))
        torch.nn.init.uniform_(self.know_params, -1.0, 1.0)

        self.out = nn.Sequential(
            nn.Linear(d_model * 2, d_fc),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_fc, d_fc // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_fc // 2, 1),
        )

        if proj:
            self.proj = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU())
        else:
            self.proj = None

        self.dropout_rate = dropout
        self.lambda_cl = lambda_cl
        self.hard_neg = hard_neg
        self.shortcut = shortcut
        self.n_layers = n_layers
        self.window = window

        # 位置编码（sinusoidal，不可学习）
        self._build_positional_encoding(d_model, max_len=512)

    def _build_positional_encoding(self, d_model, max_len=512):
        """构建 sinusoidal 位置编码 (Vaswani et al., 2017)"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_pe', pe)  # (max_len, d_model)

    def _add_positional_encoding(self, x):
        """给序列添加位置编码, x: (batch, seq_len, d_model)"""
        seq_len = x.size(1)
        return x + self.pos_pe[:seq_len].unsqueeze(0)

    def _compute_repeat_counts(self, q):
        """计算每个位置题目的重复出现次数（遗忘信号）。

        q: (batch, seq_len)，题目ID序列（已将负值填充为0）
        返回: (batch, seq_len) 的 long tensor，每个位置记录该题目前面已出现几次+1
        """
        B, L = q.size()
        counts = torch.ones(B, L, device=q.device, dtype=torch.long)
        for b in range(B):
            seen = {}
            for t in range(L):
                qid = q[b, t].item()
                counts[b, t] = seen.get(qid, 0) + 1
                seen[qid] = counts[b, t]
        return counts

    def forward(self, q_emb, s_emb, lens, need_scores=False):
        if self.shortcut:
            hq, _ = self.block1(q_emb, q_emb, q_emb, lens, peek_cur=True, need_scores=need_scores)
            hs, scores = self.block2(s_emb, s_emb, s_emb, lens, peek_cur=True, need_scores=need_scores)
            z, k_scores = self.block3(hq, hq, hs, lens, peek_cur=False, need_scores=need_scores)
            return z, scores, k_scores

        if self.n_layers == 1:
            hq = q_emb
            p, q_scores = self.block1(q_emb, q_emb, s_emb, lens, peek_cur=True, need_scores=need_scores)
        elif self.n_layers == 2:
            hq = q_emb
            hs, _ = self.block1(s_emb, s_emb, s_emb, lens, peek_cur=True, need_scores=need_scores)
            p, q_scores = self.block2(hq, hq, hs, lens, peek_cur=True, need_scores=need_scores)
        else:
            hq, _ = self.block1(q_emb, q_emb, q_emb, lens, peek_cur=True, need_scores=need_scores)
            hs, _ = self.block2(s_emb, s_emb, s_emb, lens, peek_cur=True, need_scores=need_scores)
            p, q_scores = self.block3(hq, hq, hs, lens, peek_cur=True, need_scores=need_scores)

        bs, seqlen, d_model = p.size()
        n_know = self.n_know

        query = (
            self.know_params[None, :, None, :]
            .expand(bs, -1, seqlen, -1)
            .contiguous()
            .view(bs * n_know, seqlen, d_model)
        )
        hq = hq.unsqueeze(1).expand(-1, n_know, -1, -1).reshape_as(query)
        p = p.unsqueeze(1).expand(-1, n_know, -1, -1).reshape_as(query)

        z, k_scores = self.block4(
            query, hq, p, torch.repeat_interleave(lens, n_know), peek_cur=False, need_scores=need_scores
        )
        z = (
            z.view(bs, n_know, seqlen, d_model)
            .transpose(1, 2)
            .contiguous()
            .view(bs, seqlen, -1)
        )
        if need_scores and k_scores is not None:
            k_scores = (
                k_scores.view(bs, n_know, self.n_heads, seqlen, seqlen)
                .permute(0, 2, 3, 1, 4)
                .contiguous()
            )
        else:
            k_scores = None
        return z, q_scores, k_scores

    def embedding(self, q, s, pid=None, kc_ids=None, edge_index=None, 
                  q_text_input=None, kc_text_input=None):
        """
        增强的 embedding 函数（整合语义增强与图结构增强）
        
        语义增强:
            基础形式: q_emb = self.q_embed(q_id)      # 仅 ID embedding
            当前形式: q_text_emb + proj(kc_emb)       # 语义 + 知识点融合
        
        图结构增强:
            基础输入: input_emb = q_emb + time_emb
            当前输入: input_emb = q_emb + prereq_emb + time_emb
        """
        lens = (s >= 0).sum(dim=1)
        q = q.masked_fill(q < 0, 0)
        s = s.masked_fill(s < 0, 0)
        # s_embed 只有两个槽位(0/1)，这里做硬边界以避免索引越界
        s = s.clamp(min=0, max=1)

        # ============ 修改点1: LLM语义嵌入 ============
        if self.use_llm and self.use_precomputed_llm:
            q_emb = self.q_embed(q, kc_ids)
        elif self.use_llm and q_text_input is not None:
            # 使用LLMGroundingWithID
            q_emb = self.q_embed(q, q_text_input, kc_text_input)
        elif self.use_llm:
            # LLM 全部不可用，退化为 ID 投影（128→256）
            id_emb = self.q_embed.q_embed(q)  # [B, L, 128]
            q_emb = self.q_embed.id_to_llm_proj(id_emb)  # [B, L, 256]
        else:
            # 纯 ID 嵌入
            q_emb = self.q_embed(q)

        # ============ 修改点2: 加入GNN先决图嵌入 ============
        if self.use_gnn and kc_ids is not None and edge_index is not None:
            prereq_emb = self.gnn(edge_index, kc_ids)
            q_emb = q_emb + prereq_emb

        # 遗忘机制：重复次数嵌入（q 中同一题目出现的累计次数）
        repeat_counts = self._compute_repeat_counts(q)  # (B, L)
        repeat_emb = self.repeat_embed(repeat_counts.clamp(max=self.max_repeats - 1))
        q_emb = q_emb + repeat_emb

        # s_emb 在所有 q_emb 增强完成后构造，确保包含 GNN 和 repeat 特征
        s_emb = self.s_embed(s) + q_emb

        p_diff = 0.0

        if pid is not None:
            pid = pid.masked_fill(pid < 0, 0)
            p_diff = self.p_diff_embed(pid)

            q_diff_emb = self.q_diff_embed(q)
            q_emb += q_diff_emb * p_diff

            s_diff_emb = self.s_diff_embed(s) + q_diff_emb
            s_emb += s_diff_emb * p_diff

        # Embedding dropout（正则化，仅训练时）
        if self.training:
            q_emb = self.emb_dropout(q_emb)
            s_emb = self.emb_dropout(s_emb)

        return q_emb, s_emb, lens, p_diff

    def readout(self, z, query):
        bs, seqlen, _ = query.size()
        key = (
            self.know_params[None, None, :, :]
            .expand(bs, seqlen, -1, -1)
            .view(bs * seqlen, self.n_know, -1)
        )
        value = z.reshape(bs * seqlen, self.n_know, -1)

        beta = torch.matmul(
            key,
            query.reshape(bs * seqlen, -1, 1),
        ).view(bs * seqlen, 1, self.n_know)
        alpha = torch.softmax(beta, -1)
        return torch.matmul(alpha, value).view(bs, seqlen, -1)

    def predict(self, q, s, pid=None, kc_ids=None, edge_index=None,
                q_text_input=None, kc_text_input=None, n=1, need_scores=False):
        """
        预测函数（支持LLM文本输入）
        
        Args:
            q: 题目ID序列
            s: 作答序列
            pid: 题目ID（可选）
            kc_ids: 知识点ID
            edge_index: 先决图边索引
            q_text_input: 题目文本BERT输入（修改点1）
            kc_text_input: 知识点文本BERT输入（修改点1）
            n: 预测窗口
        """
        q_emb, s_emb, lens, p_diff = self.embedding(
            q, s, pid, kc_ids, edge_index, q_text_input, kc_text_input
        )
        z, q_scores, k_scores = self(q_emb, s_emb, lens, need_scores=need_scores)

        if self.shortcut:
            assert n == 1, "AKT does not support T+N prediction"
            h = z
        else:
            query = q_emb[:, n - 1 :, :]
            h = self.readout(z[:, : query.size(1), :], query)

        y = self.out(torch.cat([query, h], dim=-1)).squeeze(-1)

        if pid is not None:
            return y, z, q_emb, (p_diff**2).mean() * 1e-3, (q_scores, k_scores)
        else:
            return y, z, q_emb, 0.0, (q_scores, k_scores)

    def weighted_bce_loss(self, logits, targets, neg_weight=1.0):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        weights = targets + (1.0 - targets) * neg_weight
        weighted_bce_loss = weights * bce_loss
        return weighted_bce_loss.mean()

    def efficient_cosine_similarity(self, z):
        z_norm = F.normalize(z, p=2, dim=1)
        return torch.mm(z_norm, z_norm.transpose(0, 1))

    def knowledge_consistency_loss(self, z, lens=None, consistency_weight=0.5):
        batch_size, seq_len, feature_size = z.size()

        if lens is not None:
            # 只用真实序列长度内的位置，过滤 padding
            valid_z = []
            for b in range(batch_size):
                valid_z.append(z[b, :lens[b], :])
            z_valid = torch.cat(valid_z, dim=0)  # (sum(lens), feature_size)
            n = z_valid.size(0)
            if n < 2:
                return torch.tensor(0.0, device=z.device)
            z_norm = F.normalize(z_valid, p=2, dim=1)
            sim_matrix = torch.mm(z_norm, z_norm.transpose(0, 1))
            mask = torch.eye(n, device=z.device)
            loss = (sim_matrix - mask).pow(2).sum() / (n * (n - 1))
        else:
            z_viewed = z.view(-1, feature_size)
            n = z_viewed.size(0)
            z_norm = F.normalize(z_viewed, p=2, dim=1)
            sim_matrix = torch.mm(z_norm, z_norm.transpose(0, 1))
            mask = torch.eye(n, device=z.device)
            loss = (sim_matrix - mask).pow(2).sum() / (n * (n - 1))

        return consistency_weight * loss

    def compute_embedding_contrastive_loss(self, q, kc_ids=None):
        """
        v3: 计算嵌入级对比损失（InfoNCE）

        在 LLM 投影空间上施加辅助对比损失，强制同 KC 题目的 LLM 投影靠近。
        """
        if not self.use_llm or not self.use_precomputed_llm:
            return torch.tensor(0.0, device=q.device)

        return self.q_embed.compute_contrastive_loss(
            q, kc_ids, self.contrast_temperature
        )

    def get_loss(self, q, s, pid=None, kc_ids=None, edge_index=None,
                 q_text_input=None, kc_text_input=None,
                 neg_weight=1.2, consistency_weight=0.05):
        """
        计算损失（支持LLM文本输入 + v3辅助对比损失）
        """
        logits, z, _, reg_loss, _ = self.predict(
            q, s, pid, kc_ids, edge_index, q_text_input, kc_text_input, need_scores=False
        )
        masked_labels = s[s >= 0].float()
        masked_logits = logits[s >= 0]

        bce_loss = self.weighted_bce_loss(masked_logits, masked_labels, neg_weight)
        lens = (s >= 0).sum(dim=1)
        cons_loss = self.knowledge_consistency_loss(z, lens=lens, consistency_weight=consistency_weight)

        total_loss = bce_loss + cons_loss + reg_loss

        # v3: 辅助嵌入对比损失
        if self.use_llm and self.lambda_contra > 0:
            contra_loss = self.compute_embedding_contrastive_loss(q, kc_ids)
            total_loss = total_loss + self.lambda_contra * contra_loss

        return total_loss

    def get_cl_loss(self, q, s, pid=None, kc_ids=None, edge_index=None,
                    q_text_input=None, kc_text_input=None):
        """
        计算对比学习损失（支持LLM文本输入）
        """
        bs = s.size(0)

        lens = (s >= 0).sum(dim=1)
        minlen = lens.min().item()
        if minlen < MIN_SEQ_LEN:
            return self.get_loss(q, s, pid, kc_ids, edge_index, q_text_input, kc_text_input)

        q_ = q.clone()
        s_ = s.clone()

        if pid is not None:
            pid_ = pid.clone()
        else:
            pid_ = None

        for b in range(bs):
            idx = random.sample(
                range(lens[b] - 1), max(1, int(lens[b] * self.dropout_rate))
            )
            for i in idx:
                q_[b, i], q_[b, i + 1] = q_[b, i + 1], q_[b, i]
                s_[b, i], s_[b, i + 1] = s_[b, i + 1], s_[b, i]
                if pid_ is not None:
                    pid_[b, i], pid_[b, i + 1] = pid_[b, i + 1], pid_[b, i]

        s_flip = s.clone() if self.hard_neg else s_
        for b in range(bs):
            idx = random.sample(
                range(lens[b]), max(1, int(lens[b] * self.dropout_rate))
            )
            for i in idx:
                s_flip[b, i] = 1 - s_flip[b, i]
        s_flip = s_flip.masked_fill(s_flip < 0, 0).clamp(min=0, max=1)
        if not self.hard_neg:
            s_ = s_flip

        logits, z_1, q_emb, reg_loss, _ = self.predict(
            q, s, pid, kc_ids, edge_index, q_text_input, kc_text_input, need_scores=False
        )
        masked_logits = logits[s >= 0]

        _, z_2, *_ = self.predict(
            q_, s_, pid_, kc_ids, edge_index, q_text_input, kc_text_input, need_scores=False
        )

        if self.hard_neg:
            _, z_3, *_ = self.predict(
                q, s_flip, pid, kc_ids, edge_index, q_text_input, kc_text_input, need_scores=False
            )

        input_sim = self.sim(z_1[:, :minlen, :], z_2[:, :minlen, :])
        if self.hard_neg:
            hard_neg = self.sim(z_1[:, :minlen, :], z_3[:, :minlen, :])
            input_sim = torch.cat([input_sim, hard_neg], dim=1)
        target = (
            torch.arange(s.size(0))[:, None]
            .to(self.know_params.device)
            .expand(-1, minlen)
        )
        cl_loss = F.cross_entropy(input_sim, target)

        masked_labels = s[s >= 0].float()
        pred_loss = F.binary_cross_entropy_with_logits(
            masked_logits, masked_labels, reduction="mean"
        )

        for i in range(1, self.window):
            label = s[:, i:]
            query = q_emb[:, i:, :]
            h = self.readout(z_1[:, : query.size(1), :], query)
            y = self.out(torch.cat([query, h], dim=-1)).squeeze(-1)

            pred_loss += F.binary_cross_entropy_with_logits(
                y[label >= 0], label[label >= 0].float()
            )
        pred_loss /= self.window

        cons_loss = self.knowledge_consistency_loss(
            z_1, lens=lens, consistency_weight=0.05
        )

        total = pred_loss + cl_loss * self.lambda_cl + cons_loss + reg_loss

        # v3: 辅助嵌入对比损失
        if self.use_llm and self.lambda_contra > 0:
            contra_loss = self.compute_embedding_contrastive_loss(q, kc_ids)
            total = total + self.lambda_contra * contra_loss

        return total, pred_loss, cl_loss

    def sim(self, z1, z2):
        bs, seqlen, _ = z1.size()
        z1 = z1.unsqueeze(1).view(bs, 1, seqlen, self.n_know, -1)
        z2 = z2.unsqueeze(0).view(1, bs, seqlen, self.n_know, -1)
        if self.proj is not None:
            z1 = self.proj(z1)
            z2 = self.proj(z2)
        return F.cosine_similarity(z1.mean(-2), z2.mean(-2), dim=-1) / 0.05

    def tracing(self, q, s, pid=None, kc_ids=None, edge_index=None):
        pad = torch.tensor([0]).to(self.know_params.device)
        q = torch.cat([q, pad], dim=0).unsqueeze(0)
        s = torch.cat([s, pad], dim=0).unsqueeze(0)
        if pid is not None:
            pid = torch.cat([pid, pad], dim=0).unsqueeze(0)

        with torch.no_grad():
            q_emb, s_emb, lens, _ = self.embedding(q, s, pid, kc_ids, edge_index)
            z, _, _ = self(q_emb, s_emb, lens, need_scores=False)
            query = self.know_params.unsqueeze(1).expand(-1, z.size(1), -1).contiguous()
            z = z.expand(self.n_know, -1, -1).contiguous()
            h = self.readout(z, query)
            y = self.out(torch.cat([query, h], dim=-1)).squeeze(-1)
            y = torch.sigmoid(y)

        return y
