import json
import os
import subprocess
import inspect
from datetime import datetime

import torch
import torch.optim as optim
from tqdm import tqdm

from DTransformer.eval import Evaluator
from utils.data_pipeline import add_kc_ids_to_batch, prepare_bert_inputs


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _move_optional_tensor(value, device):
    if value is None:
        return None
    return value.to(device) if isinstance(value, torch.Tensor) else value


def _call_model_method(method, q, s, pid=None, kc_ids=None, edge_index=None, q_text=None, seq_len=None):
    """按模型方法签名裁剪参数，兼容官方/包装后的基线接口。"""
    parameter_names = inspect.signature(method).parameters
    owner = getattr(method, "__self__", None)
    supports_pid = True if owner is None else getattr(owner, "_supports_pid", True)
    kwargs = {}

    if "pid" in parameter_names:
        kwargs["pid"] = pid if supports_pid else None
    if "kc_ids" in parameter_names:
        kwargs["kc_ids"] = kc_ids
    if "edge_index" in parameter_names:
        kwargs["edge_index"] = edge_index
    if "q_text" in parameter_names:
        kwargs["q_text"] = q_text
    if "seq_len" in parameter_names:
        kwargs["seq_len"] = seq_len
    if "n" in parameter_names:
        kwargs["n"] = 1

    return method(q, s, **kwargs)


def _extract_logits(prediction_output, reference=None):
    """兼容 baseline predict 的张量/元组两种返回形式。"""
    logits = prediction_output[0] if isinstance(prediction_output, (tuple, list)) else prediction_output
    if isinstance(logits, torch.Tensor) and reference is not None:
        if logits.dim() == 1 and reference.dim() == 2:
            logits = logits.unsqueeze(0)
    return logits


def _to_jsonable(value):
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(payload), f, indent=2, ensure_ascii=False)


def _read_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _slugify_tag(value):
    if not value:
        return "default"
    safe = []
    for ch in str(value):
        if ch.isalnum() or ch in {"-", "_", "."}:
            safe.append(ch)
        else:
            safe.append("-")
    tag = "".join(safe).strip("-_.")
    return tag or "default"


def resolve_output_paths(output_dir):
    artifacts_dir = os.path.join(output_dir, "artifacts")
    metrics_dir = os.path.join(output_dir, "metrics")
    meta_dir = os.path.join(output_dir, "meta")
    return {
        "root": output_dir,
        "artifacts_dir": artifacts_dir,
        "metrics_dir": metrics_dir,
        "meta_dir": meta_dir,
        "best_model": os.path.join(artifacts_dir, "best_model.pt"),
        "metrics_history": os.path.join(metrics_dir, "metrics_history.json"),
        "summary": os.path.join(metrics_dir, "summary.json"),
        "config": os.path.join(meta_dir, "config.json"),
        "split_info": os.path.join(meta_dir, "split_info.json"),
        "run_info": os.path.join(meta_dir, "run_info.json"),
    }


def update_run_status(output_paths, status, **extra_fields):
    if output_paths is None:
        return
    run_info_path = output_paths["run_info"]
    payload = _read_json(run_info_path)
    payload["status"] = status
    payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
    payload.update(_to_jsonable(extra_fields))
    if status in {"completed", "failed", "interrupted"} and not payload.get("ended_at"):
        payload["ended_at"] = payload["updated_at"]
    _write_json(run_info_path, payload)


def _resolve_git_revision(project_root):
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=project_root,
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )
    except Exception:
        commit = None
    try:
        branch = (
            subprocess.check_output(
                ["git", "branch", "--show-current"],
                cwd=project_root,
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )
    except Exception:
        branch = None
    return commit, branch


def build_batch_views(batch, config, q_texts=None, tokenizer=None, q_to_kc_mapping=None):
    has_pid = hasattr(batch, "field_index") and "pid" in getattr(batch, "field_index", {})
    if has_pid:
        q, s, pid = batch.get("q", "s", "pid")
    else:
        q, s = batch.get("q", "s")
        pid = None

    seq_len = config.get("seq_len", None)
    q_text_input = None
    if config.get("use_llm", False) and q_texts is not None:
        q_text_input = prepare_bert_inputs(q_texts, q, tokenizer)

    if seq_len is None:
        q, s, pid = [q], [s], [pid]
        if q_text_input is not None:
            q_text_input = [q_text_input]

    kc_ids = None
    need_kc_ids = config.get("use_gnn", False) or config.get("use_llm", False)
    if need_kc_ids and q_to_kc_mapping is not None:
        n_kc = config.get("gnn_n_kc", config.get("n_kc", None))
        kc_ids = add_kc_ids_to_batch(q, q_to_kc_mapping, n_kc)

    views = []
    for idx, (q_item, s_item, pid_item) in enumerate(zip(q, s, pid)):
        current_q_text = None
        if q_text_input is not None:
            current_q_text = q_text_input[idx] if len(q_text_input) > 1 else q_text_input[0]

        current_kc_ids = None
        if kc_ids is not None:
            if isinstance(kc_ids, list):
                current_kc_ids = kc_ids[idx] if len(kc_ids) > idx else kc_ids[0]
            else:
                current_kc_ids = kc_ids
            if not isinstance(current_kc_ids, torch.Tensor) and isinstance(current_kc_ids, list):
                current_kc_ids = (
                    current_kc_ids[idx] if len(current_kc_ids) > idx else current_kc_ids[0]
                )

        views.append(
            {
                "q": q_item,
                "s": s_item,
                "pid": pid_item,
                "q_text": current_q_text,
                "kc_ids": current_kc_ids,
            }
        )

    return views


def move_batch_view_to_device(batch_view, device):
    q = batch_view["q"].to(device)
    s = batch_view["s"].to(device)
    pid = _move_optional_tensor(batch_view["pid"], device)

    q_text = batch_view["q_text"]
    if q_text is not None:
        q_text = {key: value.to(device) for key, value in q_text.items()}

    kc_ids = _move_optional_tensor(batch_view["kc_ids"], device)

    return q, s, pid, kc_ids, q_text


def train_epoch(
    model,
    train_data,
    optimizer,
    config,
    q_texts=None,
    tokenizer=None,
    q_to_kc_mapping=None,
    edge_index=None,
    scaler=None,
    use_amp=False,
):
    model.train()
    device = torch.device(config.get("device", config.get("training_device", "cuda")))
    gradient_accumulation_steps = config.get(
        "gradient_accumulation_steps",
        config.get("training_gradient_accumulation_steps", 1),
    )
    use_cl_loss = config.get("cl_loss", config.get("recommendation_cl_loss", False))

    total_loss = 0.0
    total_cnt = 0
    accumulation_counter = 0
    optimizer.zero_grad()
    params_to_clip = _unwrap_model(model).parameters()

    total_batches = len(train_data.loader)
    pbar = tqdm(train_data, total=total_batches, desc="训练")

    for batch in pbar:
        batch_views = build_batch_views(
            batch,
            config,
            q_texts=q_texts,
            tokenizer=tokenizer,
            q_to_kc_mapping=q_to_kc_mapping,
        )

        for batch_view in batch_views:
            q, s, pid, current_kc_ids, current_q_text = move_batch_view_to_device(batch_view, device)
            model_to_use = _unwrap_model(model)

            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                if use_cl_loss:
                    loss_out = _call_model_method(
                        model_to_use.get_cl_loss,
                        q,
                        s,
                        pid=pid,
                        kc_ids=current_kc_ids,
                        edge_index=edge_index,
                        q_text=current_q_text,
                        seq_len=None,
                    )
                    loss = loss_out[0] if isinstance(loss_out, tuple) else loss_out
                else:
                    loss = _call_model_method(
                        model_to_use.get_loss,
                        q,
                        s,
                        pid=pid,
                        kc_ids=current_kc_ids,
                        edge_index=edge_index,
                        q_text=current_q_text,
                        seq_len=None,
                    )

            loss = loss / gradient_accumulation_steps
            accumulation_counter += 1

            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if accumulation_counter % gradient_accumulation_steps == 0:
                if use_amp and scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params_to_clip, 1.0)
                if use_amp and scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * gradient_accumulation_steps
            total_cnt += 1
            pbar.set_postfix({"loss": total_loss / total_cnt})

    if (accumulation_counter % gradient_accumulation_steps) != 0:
        if use_amp and scaler is not None:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params_to_clip, 1.0)
        if use_amp and scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

    return total_loss / total_cnt


def validate(
    model,
    valid_data,
    config,
    q_texts=None,
    tokenizer=None,
    q_to_kc_mapping=None,
    edge_index=None,
    use_amp=False,
):
    model.eval()
    device = torch.device(config.get("device", config.get("training_device", "cuda")))
    evaluator = Evaluator()

    with torch.no_grad():
        total_batches = len(valid_data.loader)
        for batch in tqdm(valid_data, total=total_batches, desc="验证"):
            batch_views = build_batch_views(
                batch,
                config,
                q_texts=q_texts,
                tokenizer=tokenizer,
                q_to_kc_mapping=q_to_kc_mapping,
            )
            for batch_view in batch_views:
                q, s, pid, current_kc_ids, current_q_text = move_batch_view_to_device(batch_view, device)
                model_to_use = _unwrap_model(model)
                eval_shift = int(getattr(model_to_use, "_eval_shift", 0))
                with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                    prediction_output = _call_model_method(
                        model_to_use.predict,
                        q,
                        s,
                        pid=pid,
                        kc_ids=current_kc_ids,
                        edge_index=edge_index,
                        q_text=current_q_text,
                        seq_len=None,
                    )
                y = _extract_logits(prediction_output, reference=s)
                if eval_shift > 0:
                    evaluator.evaluate(s[:, eval_shift:], torch.sigmoid(y))
                else:
                    evaluator.evaluate(s, torch.sigmoid(y))

    return evaluator.report()


def build_optimizer_and_scheduler(model, config):
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.get("learning_rate", 0.001),
        weight_decay=config.get("training_l2", config.get("l2", 1e-5)),
    )
    epochs = config.get("epochs", config.get("n_epochs", 30))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=config.get("learning_rate", 0.001) * 0.01,
    )
    return optimizer, scheduler


def select_runtime_device(config):
    device = torch.device(config.get("device", config.get("training_device", "cuda")))
    gpu_device_ids = config.get("gpu_device_ids", None)

    if gpu_device_ids and str(device).startswith("cuda") and torch.cuda.is_available():
        gpu_free = []
        for gid in gpu_device_ids:
            if gid < torch.cuda.device_count():
                free_mem = torch.cuda.mem_get_info(gid)[0]
                gpu_free.append((gid, free_mem))
        gpu_free.sort(key=lambda item: -item[1])

        if gpu_free:
            best_gpu = gpu_free[0]
            selected = torch.device(f"cuda:{best_gpu[0]}")
            print(
                f"✅ 自动选择 GPU {best_gpu[0]}（空闲 {best_gpu[1] // 1024} MiB / "
                f"{torch.cuda.get_device_properties(best_gpu[0]).total_memory // 1024} MiB）"
            )
            if len(gpu_free) > 1:
                print(f"   其他 GPU: {[(gid, free_mem // 1024) for gid, free_mem in gpu_free[1:]]}")
            return selected

        print("⚠️  配置的 GPU 不可用，回退 CPU")
        return torch.device("cpu")

    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    print(f"📱 使用设备: {device}")
    return device


def initialize_runtime(model, config, baseline_name=None, edge_index=None):
    device = select_runtime_device(config)
    config["device"] = str(device)
    config["training_device"] = str(device)

    if str(device).startswith("cuda"):
        torch.cuda.set_device(device)

    model.to(device)
    if edge_index is not None:
        edge_index = edge_index.to(device)

    amp_flag = config.get("use_amp", config.get("training_use_amp", True))
    use_amp = bool(amp_flag) and str(device).startswith("cuda")
    if baseline_name:
        use_amp = False
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"⚙️  AMP混合精度: {'开启' if use_amp else '关闭'}")

    return device, edge_index, scaler, use_amp


def create_output_dir(project_root, mode, dataset_name, config, split_info):
    if not config.get("training_save_model", config.get("save_model", True)):
        return None, None

    now = datetime.now()
    date_part = now.strftime("%Y-%m-%d")
    time_part = now.strftime("%H%M%S")
    model_tag = _slugify_tag(
        os.path.basename(config.get("pretrained_model", "default"))
        if config.get("use_llm", False)
        else "no-llm"
    )
    run_tag = f"{time_part}_{model_tag}"
    output_dir = os.path.join(
        project_root,
        "output",
        "runs",
        dataset_name,
        mode,
        date_part,
        run_tag,
    )
    output_paths = resolve_output_paths(output_dir)
    for key in ("root", "artifacts_dir", "metrics_dir", "meta_dir"):
        os.makedirs(output_paths[key], exist_ok=True)

    _write_json(output_paths["config"], config)
    _write_json(output_paths["split_info"], split_info)
    git_commit, git_branch = _resolve_git_revision(project_root)
    _write_json(
        output_paths["run_info"],
        {
            "run_id": f"{dataset_name}/{mode}/{date_part}/{run_tag}",
            "dataset": dataset_name,
            "mode": mode,
            "date": date_part,
            "run_tag": run_tag,
            "status": "running",
            "started_at": now.isoformat(timespec="seconds"),
            "updated_at": now.isoformat(timespec="seconds"),
            "git_commit": git_commit,
            "git_branch": git_branch,
            "output_dir": os.path.relpath(output_dir, project_root),
        },
    )

    print(f"📁 输出目录: {output_dir}")
    return output_dir, output_paths


def save_metrics_history(history_path, history):
    if history_path is None:
        return
    _write_json(history_path, history)


def save_best_model(model, output_dir):
    if output_dir is None:
        return None

    model_path = resolve_output_paths(output_dir)["best_model"]
    state_dict = _unwrap_model(model).state_dict()
    torch.save(state_dict, model_path)
    print(f"💾 保存最佳模型: {model_path}")
    return model_path


def load_best_model_if_available(model, output_dir, device):
    if output_dir is None:
        return None

    model_path = resolve_output_paths(output_dir)["best_model"]
    if not os.path.exists(model_path):
        return None

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    print(f"\n📂 加载最佳模型: {model_path}")
    return model_path


def save_training_summary(summary_path, best_epoch, best, test_results, split_info):
    if summary_path is None:
        return
    _write_json(
        summary_path,
        {
            "best_epoch": best_epoch,
            "best_valid": best,
            "test": test_results,
            "split_info": split_info,
        },
    )
