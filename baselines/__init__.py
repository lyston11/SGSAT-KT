import torch
import torch.nn as nn
import torch.nn.functional as F
from pykt.models.init_model import init_model

from .dtransformer import DTransformer as OfficialDTransformer


class PyKTBaselineWrapper(nn.Module):
    """适配 pykt 基线模型到当前训练管线接口。"""

    def __init__(self, model, model_name, num_c, use_pid=False):
        super().__init__()
        self.model = model
        self.model_name = model_name
        self.num_c = num_c
        self.use_pid = use_pid
        self._is_baseline = True
        self._eval_shift = 1

    def _prepare_inputs(self, q, s, pid=None):
        valid_mask = (q[:, 1:] >= 0) & (s[:, 1:] >= 0)
        q = q.masked_fill(q < 0, 0).long()
        s = s.masked_fill(s < 0, 0).long()
        pid = pid.masked_fill(pid < 0, 0).long() if pid is not None else None
        return q, s, pid, valid_mask

    def _predict_probabilities(self, q, s, pid=None):
        q, s, pid, valid_mask = self._prepare_inputs(q, s, pid)
        reg_loss = 0.0

        if self.model_name == "dkt":
            probs = self.model(q[:, :-1], s[:, :-1])
            target_q = q[:, 1:]
            probs = (probs * F.one_hot(target_q, self.num_c).float()).sum(-1)
        elif self.model_name == "dkvmn":
            probs = self.model(q, s)[:, 1:]
        elif self.model_name == "sakt":
            probs = self.model(q[:, :-1], s[:, :-1], q[:, 1:])
        elif self.model_name == "akt":
            item_seq = pid if self.use_pid and pid is not None else q
            probs, reg_loss = self.model(q, s, item_seq)
            probs = probs[:, 1:]
        else:
            raise ValueError(f"未知 pykt 基线模型: {self.model_name}")

        return probs, valid_mask, reg_loss

    def get_loss(
        self,
        q,
        s,
        pid=None,
        kc_ids=None,
        edge_index=None,
        q_text=None,
        seq_len=None,
    ):
        probs, valid_mask, reg_loss = self._predict_probabilities(q, s, pid)
        if valid_mask.sum() == 0:
            loss = probs.sum() * 0
        else:
            preds = probs[valid_mask].clamp(1e-6, 1 - 1e-6)
            labels = s[:, 1:][valid_mask].float()
            loss = F.binary_cross_entropy(preds, labels)

        if isinstance(reg_loss, torch.Tensor):
            loss = loss + reg_loss
        elif reg_loss:
            loss = loss + float(reg_loss)
        return loss

    def get_cl_loss(
        self,
        q,
        s,
        pid=None,
        kc_ids=None,
        edge_index=None,
        q_text=None,
        seq_len=None,
    ):
        return self.get_loss(q, s, pid, kc_ids, edge_index, q_text, seq_len)

    def predict(
        self,
        q,
        s,
        pid=None,
        kc_ids=None,
        edge_index=None,
        q_text=None,
        seq_len=None,
    ):
        probs, _, reg_loss = self._predict_probabilities(q, s, pid)
        logits = torch.logit(probs.clamp(1e-6, 1 - 1e-6))
        return logits, None, None, reg_loss, None


def create_baseline_model(
    name,
    n_questions,
    n_pid=0,
    d_model=256,
    n_heads=8,
    n_layers=2,
    dropout=0.2,
    batch_size=16,
    seq_len=200,
    device="cpu",
):
    """工厂函数：创建 pykt 官方基线模型并包装到当前训练管线。"""

    if name == "dtransformer":
        model = OfficialDTransformer(
            n_questions,
            n_pid=n_pid,
            d_model=128,
            d_fc=256,
            n_heads=8,
            n_know=16,
            n_layers=1,
            dropout=0.05,
            lambda_cl=0.1,
            proj=False,
            hard_neg=True,
            window=1,
            shortcut=False,
        ).to(device)
        model._is_baseline = True
        model._eval_shift = 0
        return model

    emb_type = "qid"
    data_config = {
        "num_c": n_questions,
        "num_q": n_pid if n_pid > 0 else 0,
        "emb_path": "",
    }

    if name == "sakt":
        model_config = {
            "seq_len": max(1, (seq_len or 200) - 1),
            "emb_size": d_model,
            "num_attn_heads": n_heads,
            "dropout": dropout,
            "num_en": n_layers,
        }
    elif name == "akt":
        model_config = {
            "d_model": d_model,
            "n_blocks": n_layers,
            "dropout": dropout,
            "d_ff": d_model * 2,
            "num_attn_heads": n_heads,
            "l2": 1e-5,
        }
    elif name == "dkt":
        model_config = {
            "emb_size": d_model,
            "dropout": dropout,
        }
    elif name == "dkvmn":
        model_config = {
            "dim_s": d_model,
            "size_m": 50,
            "dropout": dropout,
        }
    else:
        raise ValueError(f"未知基线模型: {name}")

    model = init_model(name, model_config, data_config, emb_type)
    if model is None:
        raise RuntimeError(f"pykt 未能初始化基线模型: {name}")
    model = model.to(device)
    return PyKTBaselineWrapper(model, name, n_questions, use_pid=(name == "akt" and n_pid > 0))
