import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleGCNLayer(nn.Module):
    """简单GCN层（不依赖torch_geometric）"""

    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x, edge_index):
        if edge_index is None or edge_index.shape[1] == 0:
            return self.linear(x)
        n_nodes = x.size(0)

        edge_index = edge_index.clamp(0, n_nodes - 1)

        adj = torch.zeros(n_nodes, n_nodes, device=x.device, dtype=x.dtype)
        adj[edge_index[0], edge_index[1]] = 1.0
        degree = adj.sum(dim=1, keepdim=True).clamp(min=1)
        norm_adj = adj / degree
        x = torch.matmul(norm_adj, x)
        return self.linear(x)


class GNNPrerequisiteGraph(nn.Module):
    """GNN先决图模块（创新点2）"""

    def __init__(self, n_kc, d_model=128, n_layers=2, dropout=0.1):
        super().__init__()
        self.n_kc = n_kc
        self.kc_embed = nn.Embedding(n_kc, d_model)
        self.gcn_layers = nn.ModuleList(
            [SimpleGCNLayer(d_model) for _ in range(n_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, edge_index, kc_ids=None):
        x = self.kc_embed.weight
        if edge_index is not None:
            if edge_index.device != x.device:
                edge_index = edge_index.clone().to(x.device)
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


class DCFSimGraphEnhanced:
    """动态认知融合相似度（图增强版）— 后处理工具类"""

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
        self.user_records[user_id].append(
            {
                "question_id": question_id,
                "correct": correct,
                "difficulty": difficulty if difficulty is not None else 0.5,
            }
        )
        if difficulty is not None:
            self.question_difficulty[question_id] = difficulty
        self.similarity_matrix = None

    def compute_similarity(self, user_u, user_v, kc_mapping=None):
        if user_u not in self.user_records or user_v not in self.user_records:
            return 0.5

        q_u = set(r["question_id"] for r in self.user_records[user_u])
        q_v = set(r["question_id"] for r in self.user_records[user_v])
        common = q_u & q_v
        if not common:
            return 0.0

        cos_sim = len(common) / max(len(q_u | q_v), 1)

        diff_sim = 0.0
        if q_u and q_v:
            diff_u = [self.question_difficulty.get(q, 0.5) for q in q_u]
            diff_v = [self.question_difficulty.get(q, 0.5) for q in q_v]
            avg_diff_u = sum(diff_u) / len(diff_u)
            avg_diff_v = sum(diff_v) / len(diff_v)
            diff_sim = 1.0 - abs(avg_diff_u - avg_diff_v)

        anomaly = 0.5

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

        final_sim = (
            0.5 * cos_sim
            + 0.25 * diff_sim
            + 0.15 * anomaly
            + 0.1 * graph_sim
        )

        return final_sim

    def get_k_nearest_neighbors(self, user_id, k, kc_mapping=None):
        sims = []
        for v in range(self.n_users):
            if v != user_id:
                sim = self.compute_similarity(user_id, v, kc_mapping)
                sims.append((v, sim))
        sims.sort(key=lambda x: -x[1])
        return sims[:k]
