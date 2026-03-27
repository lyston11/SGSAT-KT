"""
预计算嵌入加载器
加载离线预计算的 Qwen 语义嵌入
"""
import os
import pickle
import numpy as np
import torch
import torch.nn as nn


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
        self.hidden_size = data.get("hidden_size", 2560)  # Qwen3 默认隐藏层大小

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
    预计算嵌入层
    替代原来的 LLMGrounding，直接加载预计算的嵌入向量
    """
    def __init__(self, precomputed_embeddings, d_model=128, use_llm=True):
        super().__init__()
        self.use_llm = use_llm
        self.d_model = d_model
        self.proj_q = None
        self.proj_kc = None
        self.W_p = None

        if use_llm and precomputed_embeddings is not None:
            # 获取隐藏层大小
            hidden_size = precomputed_embeddings.hidden_size

            # 创建投影层：将预计算的嵌入投影到 d_model 维度
            self.proj_q = nn.Linear(hidden_size, d_model)
            self.proj_kc = nn.Linear(hidden_size, d_model)
            self.W_p = nn.Linear(d_model, d_model, bias=False)

            # 保存预计算嵌入的引用
            self.precomputed = precomputed_embeddings

            print(f"✅ 使用预计算嵌入层: {hidden_size} -> {d_model}")
        else:
            self.precomputed = None
            print(f"⚠️  未使用预计算嵌入")

    def forward(self, q_ids, kc_ids=None):
        """
        前向传播

        Args:
            q_ids: 题目ID (batch, seq_len)
            kc_ids: 知识点ID (batch, seq_len) [可选]

        Returns:
            q_emb: 题目嵌入 (batch, seq_len, d_model)
        """
        if not self.use_llm or self.precomputed is None:
            # 返回零向量
            batch_size, seq_len = q_ids.shape
            out_dtype = self.proj_q.weight.dtype if self.proj_q is not None else torch.float32
            return torch.zeros(batch_size, seq_len, self.d_model,
                             device=q_ids.device, dtype=out_dtype)

        batch_size, seq_len = q_ids.shape
        q_ids_flat = q_ids.cpu().numpy().flatten().tolist()

        # 获取预计算嵌入
        q_embs_np = self.precomputed.get_batch_question_embeddings(q_ids_flat)
        q_embs_np = q_embs_np.reshape(batch_size, seq_len, -1)

        # 转换为张量
        q_embs = torch.from_numpy(q_embs_np).to(
            device=q_ids.device,
            dtype=self.proj_q.weight.dtype,
        )

        # 投影到 d_model 维度
        e_q = self.proj_q(q_embs)

        # 如果有知识点，融合知识点嵌入
        if kc_ids is not None:
            kc_ids_flat = kc_ids.cpu().numpy().flatten().tolist()
            kc_embs_np = self.precomputed.get_batch_kc_embeddings(kc_ids_flat)
            kc_embs_np = kc_embs_np.reshape(batch_size, seq_len, -1)
            kc_embs = torch.from_numpy(kc_embs_np).to(
                device=kc_ids.device,
                dtype=self.proj_kc.weight.dtype,
            )

            e_kc = self.proj_kc(kc_embs)
            e_q = e_q + self.W_p(e_kc)

        return e_q


class LLMGroundingWithPrecomputed(nn.Module):
    """
    结合ID嵌入和预计算LLM语义嵌入的混合嵌入模块
    完全替代原来的在线 LLMGrounding
    """
    def __init__(
        self,
        n_questions,
        d_model=128,
        precomputed_embeddings=None,
        use_llm=True
    ):
        super().__init__()
        self.use_llm = use_llm
        self.d_model = d_model

        # ID嵌入
        self.q_embed = nn.Embedding(n_questions + 1, d_model)

        # 预计算LLM语义嵌入
        if use_llm and precomputed_embeddings is not None:
            self.llm_layer = PrecomputedEmbeddingLayer(
                precomputed_embeddings, d_model, use_llm=True
            )
        else:
            self.llm_layer = None

    def forward(self, q_ids, kc_ids=None):
        """
        Args:
            q_ids: 题目ID (batch, seq_len)
            kc_ids: 知识点ID (batch, seq_len) [可选]
        Returns:
            q_emb: 混合嵌入 (batch, seq_len, d_model)
        """
        id_emb = self.q_embed(q_ids)

        if not self.use_llm or self.llm_layer is None:
            return id_emb

        # 获取预计算的LLM嵌入
        llm_emb = self.llm_layer(q_ids, kc_ids)

        # 混合：ID嵌入 + LLM嵌入
        combined_emb = id_emb + llm_emb

        return combined_emb
