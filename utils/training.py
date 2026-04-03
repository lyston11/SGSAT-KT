import json
import os
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


def build_batch_views(batch, config, q_texts=None, tokenizer=None, q_to_kc_mapping=None):
    if config.get("training_with_pid", config.get("with_pid", False)):
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
                    loss_out = model_to_use.get_cl_loss(
                        q, s, pid, current_kc_ids, edge_index, current_q_text, None
                    )
                    loss = loss_out[0] if isinstance(loss_out, tuple) else loss_out
                else:
                    loss = model_to_use.get_loss(
                        q, s, pid, current_kc_ids, edge_index, current_q_text, None
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
                with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                    y, *_ = model_to_use.predict(
                        q, s, pid, current_kc_ids, edge_index, current_q_text, None
                    )
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

    print(f"📱 使用设备: {device}")
    return device


def initialize_runtime(model, config, baseline_name=None, edge_index=None):
    device = select_runtime_device(config)
    config["device"] = str(device)
    config["training_device"] = str(device)

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
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(project_root, f"output/{mode}_{dataset_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(output_dir, "split_info.json"), "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)

    print(f"📁 输出目录: {output_dir}")
    return output_dir


def save_metrics_history(history_path, history):
    if history_path is None:
        return
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def save_best_model(model, output_dir):
    if output_dir is None:
        return None

    model_path = os.path.join(output_dir, "best_model.pt")
    state_dict = _unwrap_model(model).state_dict()
    torch.save(state_dict, model_path)
    print(f"💾 保存最佳模型: {model_path}")
    return model_path


def load_best_model_if_available(model, output_dir, device):
    if output_dir is None:
        return None

    model_path = os.path.join(output_dir, "best_model.pt")
    if not os.path.exists(model_path):
        return None

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    print(f"\n📂 加载最佳模型: {model_path}")
    return model_path


def save_training_summary(summary_path, best_epoch, best, test_results, split_info):
    if summary_path is None:
        return
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_epoch": best_epoch,
                "best_valid": best,
                "test": test_results,
                "split_info": split_info,
            },
            f,
            indent=2,
        )
