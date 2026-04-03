import csv
import json
import os
from collections import defaultdict

import numpy as np


def iter_kt_sequences(file_path, inputs):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        while True:
            header = f.readline()
            if not header:
                break

            header = header.strip()
            if not header:
                continue

            try:
                seq_len = int(header)
            except ValueError as exc:
                raise ValueError(f"{file_path} 中发现非法序列头: {header!r}") from exc

            raw_lines = []
            for _ in inputs:
                line = f.readline()
                if not line:
                    raise ValueError(f"{file_path} 在读取序列时意外结束")
                raw_lines.append(line.strip())

            sample = {"seq_len": seq_len}
            for field, line in zip(inputs, raw_lines):
                values = [int(x) for x in line.split(",") if x != ""]
                sample[field] = values[:seq_len]

            yield sample


def collect_observed_question_ids(paths, inputs):
    observed = set()
    for path in paths:
        if not os.path.exists(path):
            continue
        for sample in iter_kt_sequences(path, inputs):
            observed.update(sample.get("q", []))
    if not observed:
        raise ValueError("未能从序列文件中收集到任何 q_id")
    return sorted(observed)


def infer_question_ids(observed_q_ids, expected_n_questions):
    min_q = observed_q_ids[0]
    max_q = observed_q_ids[-1]

    if min_q == 0 and max_q == expected_n_questions:
        return list(range(0, expected_n_questions + 1)), "zero_based_inclusive"
    if min_q == 0 and max_q == expected_n_questions - 1:
        return list(range(0, expected_n_questions)), "zero_based_dense"
    if min_q == 1:
        upper = max(expected_n_questions, max_q)
        return list(range(1, upper + 1)), "one_based_dense"
    return observed_q_ids, "observed_only"


def build_default_q_to_kc_mapping(question_ids):
    if not question_ids:
        return {}, 0

    if question_ids[0] == 0:
        q_to_kc = {qid: qid for qid in question_ids}
        n_kc = max(question_ids) + 1
    else:
        q_to_kc = {qid: qid - 1 for qid in question_ids}
        n_kc = max(question_ids)

    return q_to_kc, n_kc


def load_existing_q_to_kc_mapping(question_text_path):
    if not os.path.exists(question_text_path):
        return None

    with open(question_text_path, "r", encoding="utf-8") as f:
        q_texts = json.load(f)

    q_to_kc = {}
    for qid_str, info in q_texts.items():
        if not isinstance(info, dict):
            continue
        skill = info.get("skill")
        if skill is None:
            continue
        try:
            q_to_kc[int(qid_str)] = int(skill)
        except (TypeError, ValueError):
            continue

    return q_to_kc or None


def load_assist17_skill_labels(data_dir, expected_count):
    raw_path = os.path.join(
        data_dir,
        "assist17",
        "raw",
        "Released Full Dataset",
        "anonymized_full_release_competition_dataset.csv",
    )
    if not os.path.exists(raw_path):
        return None

    labels = set()
    with open(raw_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            skill = (row.get("skill") or "").strip()
            if skill:
                labels.add(skill)

    labels = sorted(labels)
    return labels if len(labels) == expected_count else None


def build_text_payload(dataset_name, qid, kc_id, label=None):
    dataset_tag = dataset_name.upper()
    if label:
        question_text = f"{dataset_tag} skill slot {qid}. Label: {label}."
        kc_text = f"{dataset_tag} knowledge component {kc_id}. Label: {label}."
    else:
        question_text = f"{dataset_tag} question slot {qid}."
        kc_text = f"{dataset_tag} knowledge component {kc_id}."
    return question_text, kc_text


def save_text_data(dataset_name, question_ids, q_to_kc, n_kc, text_output_dir, raw_data_dir):
    os.makedirs(text_output_dir, exist_ok=True)

    label_vocab = None
    if dataset_name == "assist17":
        label_vocab = load_assist17_skill_labels(raw_data_dir, n_kc)

    question_texts = {}
    kc_texts = {}

    for qid in question_ids:
        kc_id = int(q_to_kc[qid])
        label = label_vocab[kc_id] if label_vocab and kc_id < len(label_vocab) else None
        question_text, kc_text = build_text_payload(dataset_name, qid, kc_id, label=label)
        question_texts[str(qid)] = {
            "text": question_text,
            "content": question_text,
            "skill": kc_id,
            "source": "synthetic",
        }
        kc_texts[str(kc_id)] = kc_text

    question_file = os.path.join(text_output_dir, f"{dataset_name}_question_texts.json")
    kc_file = os.path.join(text_output_dir, f"{dataset_name}_kc_texts.json")

    with open(question_file, "w", encoding="utf-8") as f:
        json.dump(question_texts, f, ensure_ascii=False, indent=2)
    with open(kc_file, "w", encoding="utf-8") as f:
        json.dump(kc_texts, f, ensure_ascii=False, indent=2)

    print(f"保存题目文本到: {question_file} ({len(question_texts)} 条)")
    print(f"保存知识点文本到: {kc_file} ({len(kc_texts)} 条)")


def extract_kc_info(file_path, inputs, q_to_kc, min_cooccurrence=5):
    print("提取知识点信息...")
    edge_weights = defaultdict(int)
    used_kc_ids = set()
    window_size = 5

    for sample in iter_kt_sequences(file_path, inputs):
        q_sequence = sample.get("q", [])
        kc_sequence = [q_to_kc[qid] for qid in q_sequence if qid in q_to_kc]
        used_kc_ids.update(kc_sequence)
        for i in range(len(kc_sequence)):
            for j in range(i + 1, min(i + window_size + 1, len(kc_sequence))):
                kc_a = kc_sequence[i]
                kc_b = kc_sequence[j]
                edge_weights[(kc_a, kc_b)] += 1

    n_kc = max(q_to_kc.values()) + 1 if q_to_kc else 0
    kc_ids = list(range(n_kc))
    print(f"发现 {len(used_kc_ids)} 个训练中实际出现的知识点，完整空间 n_kc={n_kc}")

    edges = [
        (src, dst)
        for (src, dst), weight in edge_weights.items()
        if weight >= min_cooccurrence
    ]
    print(f"构建了 {len(edges)} 条先决图边（共现次数 >= {min_cooccurrence}）")

    if edges:
        edge_index = np.array(edges, dtype=np.int64).T
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)

    return kc_ids, edge_index, n_kc


def save_preprocessed_data(kc_ids, edge_index, n_kc, output_dir, dataset_name, question_id_mode):
    os.makedirs(output_dir, exist_ok=True)

    kc_file = os.path.join(output_dir, f"{dataset_name}_kc_ids.npy")
    np.save(kc_file, np.array(kc_ids, dtype=np.int64))
    print(f"保存知识点ID到: {kc_file}")

    edge_file = os.path.join(output_dir, f"{dataset_name}_edge_index.npy")
    np.save(edge_file, edge_index)
    print(f"保存边索引到: {edge_file}")

    meta_file = os.path.join(output_dir, f"{dataset_name}_meta.json")
    meta = {
        "n_kc": n_kc,
        "n_edges": int(edge_index.shape[1]) if edge_index.ndim == 2 else 0,
        "kc_ids_file": kc_file,
        "edge_index_file": edge_file,
        "question_id_mode": question_id_mode,
    }
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"保存元信息到: {meta_file}")
