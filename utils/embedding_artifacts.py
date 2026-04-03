import json
import os

import numpy as np
import torch

from DTransformer.precomputed import PrecomputedEmbeddings


def load_text_data(dataset_name, data_dir):
    text_data_dir = os.path.join(data_dir, "text_data")
    q_text_path = os.path.join(text_data_dir, f"{dataset_name}_question_texts.json")

    if os.path.exists(q_text_path):
        with open(q_text_path, "r", encoding="utf-8") as f:
            q_texts = json.load(f)
        print(f"✓ 加载了 {len(q_texts)} 个题目文本")
        return q_texts
    return None


def load_q_to_kc_mapping(dataset_name, data_dir):
    text_data_path = os.path.join(
        data_dir, "text_data", f"{dataset_name}_question_texts.json"
    )

    if not os.path.exists(text_data_path):
        return None

    with open(text_data_path, "r", encoding="utf-8") as f:
        q_texts = json.load(f)

    q_to_kc = {}
    for q_id_str, q_data in q_texts.items():
        q_id = int(q_id_str)
        skill = q_data.get("skill", "-1")
        try:
            kc_id = int(skill)
            if kc_id >= 0:
                q_to_kc[q_id] = kc_id
        except Exception:
            pass

    print(f"✓ 创建了 {len(q_to_kc)} 个q_id->kc_id映射")
    return q_to_kc


def load_edge_index(dataset_name, data_dir, device):
    candidate_paths = [
        os.path.join(data_dir, "processed", f"{dataset_name}_edge_index.npy"),
        os.path.join(data_dir, "embeddings", "processed", f"{dataset_name}_edge_index.npy"),
    ]

    edge_path = next((path for path in candidate_paths if os.path.exists(path)), None)
    if edge_path is None:
        return None, None

    edge_index_np = np.load(edge_path)
    if edge_index_np.ndim != 2 or edge_index_np.shape[0] != 2:
        print(f"⚠️  edge_index格式异常: {edge_path}, shape={edge_index_np.shape}")
        return None, None

    edge_index = torch.from_numpy(edge_index_np).long().to(device)
    max_kc_id = int(edge_index_np.max())
    print(
        f"✓ 加载edge_index: {edge_path}, 边数={edge_index.size(1)}, max_kc_id={max_kc_id}"
    )
    return edge_index, max_kc_id


def resolve_precomputed_embedding_paths(data_dir, dataset_name):
    emb_dir = os.path.join(data_dir, "embeddings")
    candidates = {
        "question": [
            os.path.join(emb_dir, f"{dataset_name}_question_embeddings.pkl"),
            os.path.join(emb_dir, "question_embeddings.pkl"),
        ],
        "kc": [
            os.path.join(emb_dir, f"{dataset_name}_kc_embeddings.pkl"),
            os.path.join(emb_dir, "kc_embeddings.pkl"),
        ],
    }

    resolved = {}
    for emb_type, emb_candidates in candidates.items():
        resolved[emb_type] = next(
            (path for path in emb_candidates if os.path.exists(path)),
            emb_candidates[0],
        )
    return resolved


def load_precomputed_embeddings(data_dir, dataset_name, use_gnn=False, use_llm=False):
    emb_dir = os.path.join(data_dir, "embeddings")
    precomputed_paths = resolve_precomputed_embedding_paths(data_dir, dataset_name)
    precomputed = PrecomputedEmbeddings(embedding_dir=emb_dir, dataset_name=dataset_name)

    precomputed.load_question_embeddings(precomputed_paths["question"])
    if use_gnn or use_llm:
        try:
            precomputed.load_kc_embeddings(precomputed_paths["kc"])
        except FileNotFoundError:
            print("⚠️  未找到知识点预计算嵌入，将仅使用题目预计算嵌入")

    return precomputed


def validate_precomputed_embeddings(
    precomputed, dataset_name, dataset_config, q_to_kc_mapping=None
):
    issues = []

    q_dataset_name = getattr(precomputed, "question_dataset_name", None)
    kc_dataset_name = getattr(precomputed, "kc_dataset_name", None)
    if q_dataset_name != dataset_name:
        issues.append(
            f"题目嵌入 dataset_name={q_dataset_name!r} 与当前数据集 {dataset_name!r} 不一致"
        )
    if kc_dataset_name != dataset_name:
        issues.append(
            f"知识点嵌入 dataset_name={kc_dataset_name!r} 与当前数据集 {dataset_name!r} 不一致"
        )

    expected_question_count = dataset_config.get("n_questions")
    loaded_question_count = len(precomputed.question_id_to_idx or {})
    if (
        expected_question_count is not None
        and loaded_question_count < expected_question_count
    ):
        issues.append(
            f"题目嵌入数量不足: 期望至少 {expected_question_count}，实际 {loaded_question_count}"
        )

    if q_to_kc_mapping:
        missing_question_ids = sorted(
            set(q_to_kc_mapping.keys()) - set(precomputed.question_id_to_idx or {})
        )
        if missing_question_ids:
            sample = missing_question_ids[:10]
            issues.append(f"题目嵌入缺失 {len(missing_question_ids)} 个 q_id，例如 {sample}")

        missing_kc_ids = sorted(
            set(q_to_kc_mapping.values()) - set(precomputed.kc_id_to_idx or {})
        )
        if missing_kc_ids:
            sample = missing_kc_ids[:10]
            issues.append(f"知识点嵌入缺失 {len(missing_kc_ids)} 个 kc_id，例如 {sample}")

    return issues
