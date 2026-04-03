import os
import pickle

import numpy as np


class PrecomputedEmbeddings:
    """预计算嵌入加载器"""

    def __init__(self, embedding_dir="data/embeddings", dataset_name=None):
        self.embedding_dir = embedding_dir
        self.dataset_name = dataset_name
        self.question_embeddings = None
        self.kc_embeddings = None
        self.question_ids = None
        self.kc_ids = None
        self.question_id_to_idx = None
        self.kc_id_to_idx = None
        self.hidden_size = None
        self.model_path = None
        self.question_dataset_name = None
        self.kc_dataset_name = None

    def _resolve_embedding_path(self, embedding_type, explicit_path=None):
        if explicit_path is not None:
            return explicit_path

        candidates = []
        if self.dataset_name:
            candidates.append(
                os.path.join(self.embedding_dir, f"{self.dataset_name}_{embedding_type}_embeddings.pkl")
            )
        candidates.append(os.path.join(self.embedding_dir, f"{embedding_type}_embeddings.pkl"))

        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return candidates[0]

    def load_question_embeddings(self, path=None):
        path = self._resolve_embedding_path("question", explicit_path=path)

        if not os.path.exists(path):
            raise FileNotFoundError(f"题目嵌入文件不存在: {path}")

        print(f"📦 加载题目嵌入: {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.question_embeddings = data["embeddings"]
        self.question_ids = data["question_ids"]
        self.question_id_to_idx = {int(qid): idx for idx, qid in enumerate(self.question_ids)}
        self.hidden_size = data.get("hidden_size", 2560)
        self.model_path = data.get("model_path")
        self.question_dataset_name = data.get("dataset_name")

        print(f"✅ 加载了 {len(self.question_ids)} 个题目嵌入")
        print(f"📊 嵌入维度: {self.hidden_size}")

        return self.question_embeddings

    def load_kc_embeddings(self, path=None):
        path = self._resolve_embedding_path("kc", explicit_path=path)

        if not os.path.exists(path):
            raise FileNotFoundError(f"知识点嵌入文件不存在: {path}")

        print(f"📦 加载知识点嵌入: {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.kc_embeddings = data["embeddings"]
        self.kc_ids = data["kc_ids"]
        self.kc_id_to_idx = {int(kid): idx for idx, kid in enumerate(self.kc_ids)}
        if self.hidden_size is None:
            self.hidden_size = data.get("hidden_size", 2560)
        if self.model_path is None:
            self.model_path = data.get("model_path")
        self.kc_dataset_name = data.get("dataset_name")

        print(f"✅ 加载了 {len(self.kc_ids)} 个知识点嵌入")
        print(f"📊 嵌入维度: {self.hidden_size}")

        return self.kc_embeddings

    def get_question_embedding(self, question_id):
        if self.question_embeddings is None:
            raise ValueError("题目嵌入未加载，请先调用 load_question_embeddings()")

        idx = self.question_id_to_idx.get(int(question_id)) if self.question_id_to_idx else None
        if idx is None:
            return np.zeros(self.hidden_size, dtype=np.float32)

        return np.asarray(self.question_embeddings[idx], dtype=np.float32)

    def get_kc_embedding(self, kc_id):
        if self.kc_embeddings is None:
            raise ValueError("知识点嵌入未加载，请先调用 load_kc_embeddings()")

        idx = self.kc_id_to_idx.get(int(kc_id)) if self.kc_id_to_idx else None
        if idx is None:
            return np.zeros(self.hidden_size, dtype=np.float32)

        return np.asarray(self.kc_embeddings[idx], dtype=np.float32)

    def get_batch_question_embeddings(self, question_ids):
        return np.stack(
            [self.get_question_embedding(qid) for qid in question_ids],
            axis=0,
        ).astype(np.float32)

    def get_batch_kc_embeddings(self, kc_ids):
        return np.stack(
            [self.get_kc_embedding(kid) for kid in kc_ids],
            axis=0,
        ).astype(np.float32)
