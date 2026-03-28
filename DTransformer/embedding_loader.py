"""
预计算嵌入加载器（v2: 多层投影 + 拼接融合）
加载离线预计算的 Qwen 语义嵌入，通过多层投影保留更多语义信息，
并与 ID embedding 拼接后投影到 d_model。
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
    结合ID嵌入和预计算LLM语义嵌入的混合嵌入模块（v2: 拼接+投影融合）

    ID embedding 固定为 id_dim=128，不跟随 d_model 变化。
    LLM embedding 投影到 llm_proj_dim=256，后与 ID 拼接为 384，再投影到 d_model=256。
    """
    def __init__(
        self,
        n_questions,
        d_model=256,
        id_dim=128,
        llm_proj_dim=256,
        llm_inter_dim=512,
        precomputed_embeddings=None,
        use_llm=True
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
            # 拼接融合投影: id_dim + llm_proj_dim -> d_model
            self.fusion_proj = nn.Linear(id_dim + llm_proj_dim, d_model)
            self.fusion_norm = nn.LayerNorm(d_model)
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
            # 无 LLM 时: 投影 ID 到 d_model
            return self.id_proj(id_emb)

        # 获取预计算的 LLM 嵌入 (batch, seq_len, llm_proj_dim)
        llm_emb = self.llm_layer(q_ids, kc_ids)

        # 拼接融合: cat(id_emb, llm_emb) -> 投影到 d_model
        concat = torch.cat([id_emb, llm_emb], dim=-1)
        combined = self.fusion_norm(self.fusion_proj(concat))
        return combined
