from .AKT import AKT
from .DKT import DKT
from .DKVMN import DKVMN
from .SAKT import SAKT
from .dtransformer import DTransformer as OfficialDTransformer


def _mark_as_baseline(model, eval_shift=0, supports_pid=True):
    model._is_baseline = True
    model._eval_shift = eval_shift
    model._supports_pid = supports_pid
    return model


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
    """创建与 /home1/xmf/SATKT 同口径的本地基线模型。"""

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
        )
        return _mark_as_baseline(model.to(device), supports_pid=True)

    if name == "akt":
        model = AKT(
            n_questions,
            n_pid=n_pid,
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
        )
        return _mark_as_baseline(model.to(device), supports_pid=True)

    if name == "dkt":
        model = DKT(n_questions, d_model=d_model)
        return _mark_as_baseline(model.to(device), supports_pid=False)

    if name == "sakt":
        model = SAKT(
            n_questions,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            device=device,
        )
        return _mark_as_baseline(model.to(device), supports_pid=False)

    if name == "dkvmn":
        model = DKVMN(n_questions, batch_size=batch_size)
        return _mark_as_baseline(model.to(device), supports_pid=False)

    raise ValueError(f"未知基线模型: {name}")
