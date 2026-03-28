import torch
import torch.nn as nn


class BaselineWrapper(nn.Module):
    """适配基线模型到 DTransformer 训练管线接口。

    训练管线调用 get_loss(q, s, pid, kc_ids, edge_index, q_text, seq_len) 7参数，
    基线模型只需 get_loss(q, s, pid=None) 3参数。
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self._is_baseline = True

    def get_loss(self, q, s, pid=None, kc_ids=None, edge_index=None,
                 q_text=None, seq_len=None):
        return self.model.get_loss(q, s, pid)

    def get_cl_loss(self, q, s, pid=None, kc_ids=None, edge_index=None,
                    q_text=None, seq_len=None):
        return self.model.get_loss(q, s, pid)

    def predict(self, q, s, pid=None, kc_ids=None, edge_index=None,
                q_text=None, seq_len=None):
        result = self.model.predict(q, s, pid)
        # 统一返回: (logits, z, _, reg_loss, _)
        if isinstance(result, tuple):
            if len(result) == 2:    # DKT: (y, h), DKVMN: (logits, h)
                return result[0], None, None, 0.0, None
            elif len(result) == 3:  # AKT: (y, h, reg_loss)
                return result[0], None, None, result[2], None
        return result, None, None, 0.0, None  # SAKT: y only


def create_baseline_model(name, n_questions, d_model=256, n_heads=8,
                          dropout=0.2, batch_size=16, device='cpu'):
    """工厂函数：根据名称创建基线模型。"""
    if name == 'sakt':
        from .SAKT import SAKT
        return SAKT(n_questions, d_model=d_model, n_heads=n_heads,
                    n_layers=2, dropout=dropout, device=device)
    elif name == 'akt':
        from .AKT import AKT
        return AKT(n_questions, n_pid=0, d_model=d_model, d_fc=d_model * 2,
                   n_heads=n_heads, dropout=dropout)
    elif name == 'dkt':
        from .DKT import DKT
        return DKT(n_questions, d_model=d_model)
    elif name == 'dkvmn':
        from .DKVMN import DKVMN
        return DKVMN(n_questions, batch_size=batch_size)
    else:
        raise ValueError(f"未知基线模型: {name}")
