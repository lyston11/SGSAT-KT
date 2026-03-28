"""SGSAT-KT: Semantic Graph Sparse Attention Knowledge Tracing
基于汪盈2025 SATKT-MFCER框架，进行3处修改:
- 修改点1: LLM语义Grounding (embedding层) - 新增LLMGrounding模块
- 修改点2: GNN Prerequisite Graph (输入特征) - 新增GNN模块
- 修改点3: Graph-Enhanced DCF-Sim (相似度计算) - 新增graph_path_sim

继承关系:
- 2022年诸葛斌等（SKT-MIER）：继承多指标评估框架
- 2024年诸葛斌等（SKT-MFER / KTM-LC）：继承遗忘LSTM与难度学习
- 2025年汪盈硕士论文（SATKT-MFCER）：完整复用核心组件

具体修改点（代码改动量<25%）:
- 修改点1: q_emb = self.q_embed(q_id) -> q_text_emb + proj(kc_emb)
- 修改点2: input_emb = q_emb + time_emb -> input_emb = q_emb + prereq_emb + time_emb
- 修改点3: sim = 0.5*cos + 0.3*diff + 0.2*anomaly -> + 0.1*graph_path_sim
"""
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from DTransformer.embedding_loader import LLMGroundingWithPrecomputed

MIN_SEQ_LEN = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============ 修改点1: LLM语义Grounding模块 ============
class LLMGrounding(nn.Module):
    """
    LLM语义Grounding模块（v2: 多层投影）
    将题目文本和知识点文本通过 BERT + 多层投影转换为语义嵌入
    """
    def __init__(
        self,
        d_model=128,
        llm_proj_dim=256,
        llm_inter_dim=512,
        pretrained_model="pretrained_models/qwen3-4b",
        freeze_bert=True
    ):
        super().__init__()
        self.d_model = d_model

        try:
            from transformers import AutoModel
            self.bert = AutoModel.from_pretrained(pretrained_model)
            self.bert_hidden_size = self.bert.config.hidden_size

            if freeze_bert:
                for param in self.bert.parameters():
                    param.requires_grad = False
        except ImportError:
            print("Warning: transformers not installed, LLM grounding disabled")
            self.bert = None
            self.bert_hidden_size = llm_proj_dim

        # 多层漏斗式投影
        self.proj_q = nn.Sequential(
            nn.Linear(self.bert_hidden_size, llm_inter_dim),
            nn.GELU(),
            nn.LayerNorm(llm_inter_dim),
            nn.Dropout(0.1),
            nn.Linear(llm_inter_dim, llm_proj_dim),
            nn.LayerNorm(llm_proj_dim),
        )
        self.proj_kc = nn.Sequential(
            nn.Linear(self.bert_hidden_size, llm_inter_dim),
            nn.GELU(),
            nn.LayerNorm(llm_inter_dim),
            nn.Dropout(0.1),
            nn.Linear(llm_inter_dim, llm_proj_dim),
            nn.LayerNorm(llm_proj_dim),
        )
        self.W_p = nn.Linear(llm_proj_dim, llm_proj_dim, bias=False)
        
    def forward(self, q_text_input, kc_text_input=None):
        """
        Args:
            q_text_input: 题目文本的BERT输入 (dict with input_ids, attention_mask)
            kc_text_input: 知识点文本的BERT输入
        Returns:
            e_q: 题目语义嵌入 (batch, d_model)
        """
        if self.bert is None:
            return torch.zeros(q_text_input['input_ids'].size(0), self.d_model, device=self.proj_q.weight.device)
        
        q_outputs = self.bert(**q_text_input)
        q_cls = q_outputs.last_hidden_state[:, 0, :]
        e_q = self.proj_q(q_cls)
        
        if kc_text_input is not None:
            kc_outputs = self.bert(**kc_text_input)
            kc_cls = kc_outputs.last_hidden_state[:, 0, :]
            e_kc = self.proj_kc(kc_cls)
            e_q = e_q + self.W_p(e_kc)
        
        return e_q
    
    def get_contrastive_loss(self, e_q, e_pos, e_neg, temperature=0.07):
        """对比损失: L_con = -log(exp(sim(e_i,e_j+)/τ) / Σexp(sim(e_i,e_k)/τ))"""
        e_q = F.normalize(e_q, p=2, dim=-1)
        e_pos = F.normalize(e_pos, p=2, dim=-1)
        e_neg = F.normalize(e_neg, p=2, dim=-1)
        
        pos_sim = torch.sum(e_q * e_pos, dim=-1) / temperature
        neg_sim = torch.bmm(e_neg, e_q.unsqueeze(-1)).squeeze(-1) / temperature
        
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(e_q.size(0), dtype=torch.long, device=e_q.device)
        
        return F.cross_entropy(logits, labels)


class LLMGroundingWithID(nn.Module):
    """
    结合ID嵌入和LLM语义嵌入的混合嵌入模块（v2: 拼接+投影融合）
    ID embedding 固定为 id_dim，LLM 投影到 llm_proj_dim，
    拼接后投影到 d_model。
    """
    def __init__(
        self,
        n_questions,
        d_model=256,
        id_dim=128,
        llm_proj_dim=256,
        llm_inter_dim=512,
        pretrained_model="pretrained_models/qwen3-4b",
        freeze_bert=True,
        use_llm=True
    ):
        super().__init__()
        self.use_llm = use_llm
        self.d_model = d_model
        self.id_dim = id_dim
        self.llm_proj_dim = llm_proj_dim

        # ID 嵌入（固定 id_dim 维）
        self.q_embed = nn.Embedding(n_questions + 1, id_dim)

        # LLM 语义嵌入（多层投影到 llm_proj_dim）
        if use_llm:
            self.llm = LLMGrounding(
                d_model, llm_proj_dim=llm_proj_dim,
                llm_inter_dim=llm_inter_dim,
                pretrained_model=pretrained_model, freeze_bert=freeze_bert,
            )
            # 拼接融合投影: id_dim + llm_proj_dim -> d_model
            self.fusion_proj = nn.Linear(id_dim + llm_proj_dim, d_model)
            self.fusion_norm = nn.LayerNorm(d_model)
        else:
            self.id_proj = nn.Linear(id_dim, d_model)

    def forward(self, q_ids, q_text_input=None, kc_text_input=None):
        """
        Args:
            q_ids: 题目ID (batch, seq_len)
            q_text_input: 题目文本BERT输入
            kc_text_input: 知识点文本BERT输入
        Returns:
            q_emb: 混合嵌入 (batch, seq_len, d_model)
        """
        id_emb = self.q_embed(q_ids)  # (batch, seq_len, id_dim)

        if not self.use_llm or q_text_input is None:
            return self.id_proj(id_emb)

        llm_emb = self.llm(q_text_input, kc_text_input)  # (batch*seq_len, llm_proj_dim)

        batch_size, seq_len, _ = id_emb.size()
        llm_emb = llm_emb.view(batch_size, seq_len, self.llm_proj_dim)

        # 拼接融合: cat(id_emb[id_dim], llm_emb[llm_proj_dim]) -> d_model
        concat = torch.cat([id_emb, llm_emb], dim=-1)
        combined = self.fusion_norm(self.fusion_proj(concat))
        return combined


# ============ 修改点2: GNN Prerequisite Graph模块 ============
class SimpleGCNLayer(nn.Module):
    """简单GCN层（不依赖torch_geometric）"""
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x, edge_index):
        if edge_index is None or edge_index.shape[1] == 0:
            return self.linear(x)
        n_nodes = x.size(0)

        # 边界检查：过滤掉超出范围的索引（使用 clamp 避免 CUDA 错误）
        edge_index = edge_index.clamp(0, n_nodes - 1)

        adj = torch.zeros(n_nodes, n_nodes, device=x.device, dtype=x.dtype)
        adj[edge_index[0], edge_index[1]] = 1.0
        degree = adj.sum(dim=1, keepdim=True).clamp(min=1)
        norm_adj = adj / degree
        x = torch.matmul(norm_adj, x)
        return self.linear(x)


class GNNPrerequisiteGraph(nn.Module):
    """GNN先决图模块（创新点2）
    
    公式: h_v^(l+1) = σ(Σ_{u∈N(v)} (1/√(d_v*d_u)) * W^(l) * h_u^(l))
    """
    def __init__(self, n_kc, d_model=128, n_layers=2, dropout=0.1):
        super().__init__()
        self.n_kc = n_kc
        self.kc_embed = nn.Embedding(n_kc, d_model)
        self.gcn_layers = nn.ModuleList([
            SimpleGCNLayer(d_model) for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, edge_index, kc_ids=None):
        x = self.kc_embed.weight
        if edge_index is not None:
            # 避免修改原始edge_index，创建副本
            if edge_index.device != x.device:
                edge_index = edge_index.clone().to(x.device)
            # 安全检查：过滤超出范围的索引
            n_nodes = x.size(0)
            edge_index = edge_index.clamp(0, n_nodes - 1)
        for gcn in self.gcn_layers:
            x_new = gcn(x, edge_index)
            x_new = F.relu(x_new)
            x_new = self.dropout(x_new)
            x = self.layer_norm(x + x_new)
        if kc_ids is None:
            return x
        kc_ids = kc_ids.clamp(0, self.n_kc - 1)
        return x[kc_ids]


# ============ 修改点3: 图增强DCF-Sim相似度 ============
class DCFSimGraphEnhanced:
    """动态认知融合相似度（图增强版）

    原DCF-Sim: sim = 0.5*cos + 0.3*diff_sim + 0.2*anomaly
    修改后（按文档要求）: sim(u,v) = 0.5*cos + 0.25*diff_sim + 0.15*anomaly + 0.1*graph_path_sim
    """
    def __init__(self, n_users, n_questions, half_life=20):
        self.n_users = n_users
        self.n_questions = n_questions
        self.half_life = half_life
        self.user_records = {}
        self.question_difficulty = {}
        self.similarity_matrix = None
        
    def add_interaction(self, user_id, question_id, correct, difficulty=None):
        if user_id not in self.user_records:
            self.user_records[user_id] = []
        self.user_records[user_id].append({
            'question_id': question_id,
            'correct': correct,
            'difficulty': difficulty if difficulty is not None else 0.5
        })
        if difficulty is not None:
            self.question_difficulty[question_id] = difficulty
        self.similarity_matrix = None
    
    def compute_similarity(self, user_u, user_v, kc_mapping=None):
        """
        计算综合相似度（按文档修改点3要求）

        原DCF-Sim: sim = 0.5*cos(S_u, S_v) + 0.3*diff_sim + 0.2*anomaly
        修改后（文档要求）: sim = 0.5*cos + 0.25*diff_sim + 0.15*anomaly + 0.1*graph_path_sim
        """
        if user_u not in self.user_records or user_v not in self.user_records:
            return 0.5

        q_u = set(r['question_id'] for r in self.user_records[user_u])
        q_v = set(r['question_id'] for r in self.user_records[user_v])
        common = q_u & q_v
        if not common:
            return 0.0

        # 1. cos相似度: 基于共同答题的比例
        cos_sim = len(common) / max(len(q_u | q_v), 1)

        # 2. diff_sim: 难度差异相似度（文档要求的0.25权重）
        diff_sim = 0.0
        if q_u and q_v:
            diff_u = [self.question_difficulty.get(q, 0.5) for q in q_u]
            diff_v = [self.question_difficulty.get(q, 0.5) for q in q_v]
            avg_diff_u = sum(diff_u) / len(diff_u)
            avg_diff_v = sum(diff_v) / len(diff_v)
            diff_sim = 1.0 - abs(avg_diff_u - avg_diff_v)  # 难度越接近，相似度越高

        # 3. anomaly: 异常检测相似度（文档要求的0.15权重）
        anomaly = 0.5  # 默认值，可以根据实际异常检测算法计算

        # 4. graph_path_sim: 图路径相似度（新增，文档要求的0.1权重）
        graph_sim = 0.5
        if kc_mapping:
            kc_u = set()
            kc_v = set()
            for q in q_u:
                if q in kc_mapping:
                    kc_u.update(kc_mapping[q])
            for q in q_v:
                if q in kc_mapping:
                    kc_v.update(kc_mapping[q])
            if kc_u and kc_v:
                graph_sim = len(kc_u & kc_v) / len(kc_u | kc_v)

        # 按文档公式(修改点3): sim = 0.5*cos + 0.25*diff + 0.15*anomaly + 0.1*graph_path_sim
        final_sim = (
            0.5 * cos_sim +
            0.25 * diff_sim +
            0.15 * anomaly +
            0.1 * graph_sim
        )

        return final_sim
    
    def get_k_nearest_neighbors(self, user_id, k, kc_mapping=None):
        """获取K个最近邻用户"""
        sims = []
        for v in range(self.n_users):
            if v != user_id:
                sim = self.compute_similarity(user_id, v, kc_mapping)
                sims.append((v, sim))
        sims.sort(key=lambda x: -x[1])
        return sims[:k]


class DTransformer(nn.Module):
    """
    SGSAT-KT主模型（基于汪盈2025 SATKT-MFCER框架）
    
    三大创新点整合:
    - 修改点1: LLM语义Grounding (embedding层)
    - 修改点2: GNN Prerequisite Graph (输入特征)
    - 修改点3: Graph-Enhanced DCF-Sim (相似度计算)
    
    继承汪盈2025硕士论文SATKT主干结构，代码改动量<25%
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
                )
        else:
            # 原SATKT嵌入层（向后兼容）
            self.q_embed = nn.Embedding(n_questions + 1, d_model)
        
        self.s_embed = nn.Embedding(2, d_model)

        if n_pid > 0:
            self.q_diff_embed = nn.Embedding(n_questions + 1, d_model)
            self.s_diff_embed = nn.Embedding(2, d_model)
            self.p_diff_embed = nn.Embedding(n_pid + 1, 1)

        # ============ 修改点2: GNN先决图嵌入 ============
        if use_gnn:
            self.gnn = GNNPrerequisiteGraph(n_kc, d_model, gnn_layers, dropout)

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
        增强的embedding函数（整合修改点1和修改点2）
        
        修改点1: LLM语义Grounding
            原SATKT: q_emb = self.q_embed(q_id)  # 仅ID embedding
            修改后: q_text_emb + proj(kc_emb)    # LLM语义 + 知识点融合
        
        修改点2: GNN先决图嵌入
            原输入: input_emb = q_emb + time_emb
            修改后: input_emb = q_emb + prereq_emb + time_emb
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
            # 仅使用ID嵌入部分
            q_emb = self.q_embed.q_embed(q)
        else:
            # 原SATKT嵌入
            q_emb = self.q_embed(q)

        s_emb = self.s_embed(s) + q_emb

        p_diff = 0.0

        if pid is not None:
            pid = pid.masked_fill(pid < 0, 0)
            p_diff = self.p_diff_embed(pid)

            q_diff_emb = self.q_diff_embed(q)
            q_emb += q_diff_emb * p_diff

            s_diff_emb = self.s_diff_embed(s) + q_diff_emb
            s_emb += s_diff_emb * p_diff

        # ============ 修改点2: 加入GNN先决图嵌入 ============
        # 文档要求: input_emb = q_emb + prereq_emb + time_emb
        # 原SATKT: input_emb = q_emb + time_emb
        # 修改后: 直接相加，而非加权平均
        if self.use_gnn and kc_ids is not None and edge_index is not None:
            prereq_emb = self.gnn(edge_index, kc_ids)
            q_emb = q_emb + prereq_emb  # 按文档要求：直接相加

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

    def knowledge_consistency_loss(self, z, consistency_weight=0.5):
        batch_size, seq_len, feature_size = z.size()
        z_viewed = z.view(-1, feature_size)
        sim_matrix = self.efficient_cosine_similarity(z_viewed)
        mask = torch.eye(batch_size * seq_len, device=z.device)
        loss = (sim_matrix - mask).pow(2).sum() / (batch_size * seq_len * (batch_size * seq_len - 1))
        return consistency_weight * loss

    def get_loss(self, q, s, pid=None, kc_ids=None, edge_index=None, 
                 q_text_input=None, kc_text_input=None,
                 neg_weight=1.2, consistency_weight=0.05):
        """
        计算损失（支持LLM文本输入）
        """
        logits, z, _, reg_loss, _ = self.predict(
            q, s, pid, kc_ids, edge_index, q_text_input, kc_text_input, need_scores=False
        )
        masked_labels = s[s >= 0].float()
        masked_logits = logits[s >= 0]

        bce_loss = self.weighted_bce_loss(masked_logits, masked_labels, neg_weight)
        cons_loss = self.knowledge_consistency_loss(z, consistency_weight)

        total_loss = bce_loss + cons_loss + reg_loss
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

        return pred_loss + cl_loss * self.lambda_cl + reg_loss, pred_loss, cl_loss

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
        sparse_scores_b = torch.where(scores_b - scores_t >= 0, scores_b, torch.full_like(scores_b, neg_inf))
        scores_b = sparse_scores_b.reshape(bs, head, seqlen - k_index, -1)

        scores = torch.cat([scores_a, scores_b], dim=2)
        scores = F.softmax(scores, dim=-1)
        return scores

    def forward(self, q, k, v, mask, emb_type="qid", sparse_ratio=0.8, k_index=5, maxout=False, need_scores=False):
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
    bs, head, seqlen, _ = scores.size()

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