import torch
import torch.nn as nn
import torch.nn.functional as F


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
        freeze_bert=True,
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
        if self.bert is None:
            return torch.zeros(
                q_text_input["input_ids"].size(0),
                self.d_model,
                device=self.proj_q[0].weight.device,
            )

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
    结合ID嵌入和LLM语义嵌入的混合嵌入模块（v3: 门控融合）
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
        use_llm=True,
        id_dropout_rate=0.15,
    ):
        super().__init__()
        self.use_llm = use_llm
        self.d_model = d_model
        self.id_dim = id_dim
        self.llm_proj_dim = llm_proj_dim

        self.q_embed = nn.Embedding(n_questions + 1, id_dim)

        if use_llm:
            self.llm = LLMGrounding(
                d_model,
                llm_proj_dim=llm_proj_dim,
                llm_inter_dim=llm_inter_dim,
                pretrained_model=pretrained_model,
                freeze_bert=freeze_bert,
            )
            self.id_to_llm_proj = nn.Linear(id_dim, llm_proj_dim)
            self.gate_linear = nn.Linear(llm_proj_dim, llm_proj_dim)
            self.id_dropout = nn.Dropout(p=id_dropout_rate)
            self.fusion_norm = nn.LayerNorm(llm_proj_dim)
        else:
            self.id_proj = nn.Linear(id_dim, d_model)

    def forward(self, q_ids, q_text_input=None, kc_text_input=None):
        id_emb = self.q_embed(q_ids)

        if not self.use_llm or q_text_input is None:
            return self.id_proj(id_emb)

        if self.training:
            id_emb = self.id_dropout(id_emb)

        llm_emb = self.llm(q_text_input, kc_text_input)

        batch_size, seq_len, _ = id_emb.size()
        llm_emb = llm_emb.view(batch_size, seq_len, self.llm_proj_dim)

        id_proj = self.id_to_llm_proj(id_emb)
        gate_val = torch.sigmoid(self.gate_linear(llm_emb))
        gated_llm = gate_val * llm_emb
        combined = self.fusion_norm(gated_llm + id_proj)
        return combined
