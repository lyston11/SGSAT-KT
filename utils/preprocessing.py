import csv
import json
import os
import shutil
import tempfile
from math import ceil
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


def normalize_text(text):
    if text is None:
        return ""
    return " ".join(str(text).replace("\r", " ").replace("\n", " ").split())


def normalize_sql_scalar(value):
    if value is None:
        return ""
    return normalize_text(value)


def iter_mysql_insert_rows(sql_path, table_name):
    marker = f"INSERT INTO `{table_name}` VALUES ".encode("utf-8")
    with open(sql_path, "rb") as f:
        for raw in f:
            if not raw.startswith(marker):
                continue
            payload = raw[len(marker) :].rstrip(b";\r\n")
            if not payload:
                continue
            text = payload.decode("utf-8", errors="ignore")
            yield from parse_mysql_values_payload(text)


def parse_mysql_values_payload(payload):
    idx = 0
    payload_len = len(payload)
    while idx < payload_len:
        if payload[idx] != "(":
            idx += 1
            continue

        idx += 1
        field_chars = []
        fields = []
        in_quote = False
        escape = False

        while idx < payload_len:
            ch = payload[idx]
            if in_quote:
                if escape:
                    field_chars.append(ch)
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == "'":
                    if idx + 1 < payload_len and payload[idx + 1] == "'":
                        field_chars.append("'")
                        idx += 1
                    else:
                        in_quote = False
                else:
                    field_chars.append(ch)
            else:
                if ch == "'":
                    in_quote = True
                elif ch == ",":
                    fields.append("".join(field_chars).strip())
                    field_chars = []
                elif ch == ")":
                    fields.append("".join(field_chars).strip())
                    normalized_fields = [
                        None if field.upper() == "NULL" else field for field in fields
                    ]
                    yield normalized_fields
                    break
                else:
                    field_chars.append(ch)
            idx += 1

        idx += 1


def build_doudouyun_sql_paths(raw_root):
    sql_root = os.path.join(raw_root, "sql_dumps")
    latest_sql = os.path.join(sql_root, "app_doudouyun2_20240928.sql")
    legacy_sql = os.path.join(sql_root, "app_doudouyun2_20210401.sql")
    zjg_sql = os.path.join(sql_root, "app_zjgsujiaoxue_20240928.sql")
    return {
        "latest": latest_sql,
        "legacy": legacy_sql,
        "zjgsujiaoxue": zjg_sql,
    }


def load_doudouyun_sql_dimensions(sql_path):
    set_rows = {}
    for fields in iter_mysql_insert_rows(sql_path, "pingshifen_question_set"):
        raw_set_id = normalize_sql_scalar(fields[0])
        if not raw_set_id:
            continue
        set_rows[raw_set_id] = {
            "raw_set_id": raw_set_id,
            "teacher_id": normalize_sql_scalar(fields[1]),
            "title": normalize_sql_scalar(fields[2]),
            "count": normalize_sql_scalar(fields[3]),
            "status": normalize_sql_scalar(fields[4]),
        }

    chapter_rows = {}
    for fields in iter_mysql_insert_rows(sql_path, "pingshifen_question_chapter"):
        raw_chapter_id = normalize_sql_scalar(fields[0])
        if not raw_chapter_id:
            continue
        chapter_rows[raw_chapter_id] = {
            "raw_chapter_id": raw_chapter_id,
            "raw_set_id": normalize_sql_scalar(fields[1]),
            "title": normalize_sql_scalar(fields[2]),
            "count": normalize_sql_scalar(fields[3]),
            "status": normalize_sql_scalar(fields[4]),
        }

    bank_rows = {}
    for fields in iter_mysql_insert_rows(sql_path, "pingshifen_question_bank"):
        raw_qid = normalize_sql_scalar(fields[0])
        if not raw_qid:
            continue
        bank_rows[raw_qid] = {
            "raw_qid": raw_qid,
            "raw_set_id": normalize_sql_scalar(fields[1]),
            "raw_chapter_id": normalize_sql_scalar(fields[2]),
            "question_type": normalize_sql_scalar(fields[3]),
            "content": normalize_sql_scalar(fields[4]),
            "media_id": normalize_sql_scalar(fields[5]),
            "option_a": normalize_sql_scalar(fields[6]),
            "option_b": normalize_sql_scalar(fields[7]),
            "option_c": normalize_sql_scalar(fields[8]),
            "option_d": normalize_sql_scalar(fields[9]),
            "answer": normalize_sql_scalar(fields[10]),
            "analysis": normalize_sql_scalar(fields[11]),
            "status": normalize_sql_scalar(fields[12]),
            "gmt_create": normalize_sql_scalar(fields[13]),
            "gmt_modified": normalize_sql_scalar(fields[14]),
        }

    class_memberships = set()
    for fields in iter_mysql_insert_rows(sql_path, "pingshifen_class"):
        uid = normalize_sql_scalar(fields[1])
        cid = normalize_sql_scalar(fields[2])
        status = normalize_sql_scalar(fields[3])
        if uid and cid and status != "0":
            class_memberships.add((uid, cid))

    student_rows = {}
    for fields in iter_mysql_insert_rows(sql_path, "pingshifen_student"):
        uid = normalize_sql_scalar(fields[0])
        if not uid:
            continue
        student_rows[uid] = {
            "uid": uid,
            "name": normalize_sql_scalar(fields[2]),
            "school": normalize_sql_scalar(fields[4]),
            "status": normalize_sql_scalar(fields[17]),
        }

    exam_rows = {}
    for fields in iter_mysql_insert_rows(sql_path, "pingshifen_exam"):
        raw_mid = normalize_sql_scalar(fields[1])
        if not raw_mid:
            continue
        exam_rows[raw_mid] = {
            "raw_mid": raw_mid,
            "cid": normalize_sql_scalar(fields[2]),
            "state": normalize_sql_scalar(fields[5]),
        }

    if not bank_rows:
        raise ValueError(f"SQL 中未找到 pingshifen_question_bank: {sql_path}")

    return {
        "sets": set_rows,
        "chapters": chapter_rows,
        "bank": bank_rows,
        "class_memberships": class_memberships,
        "students": student_rows,
        "exams": exam_rows,
    }


def iter_doudouyun_sql_records(sql_path, dimensions, include_exam=True):
    bank_rows = dimensions["bank"]
    class_memberships = dimensions["class_memberships"]
    student_rows = dimensions["students"]
    exam_rows = dimensions["exams"]

    for fields in iter_mysql_insert_rows(sql_path, "pingshifen_question_record"):
        record_id = normalize_sql_scalar(fields[0])
        raw_uid = normalize_sql_scalar(fields[1])
        raw_cid = normalize_sql_scalar(fields[2])
        raw_qid = normalize_sql_scalar(fields[3])
        raw_result = normalize_sql_scalar(fields[6])
        raw_status = normalize_sql_scalar(fields[8])
        raw_timestamp = normalize_sql_scalar(fields[9])

        if raw_status != "1":
            continue
        if raw_uid not in student_rows:
            continue
        if class_memberships and (raw_uid, raw_cid) not in class_memberships:
            continue
        if raw_qid not in bank_rows or raw_result not in {"1", "2"}:
            continue
        try:
            timestamp = int(raw_timestamp)
        except ValueError:
            continue

        bank_row = bank_rows[raw_qid]
        yield {
            "record_key": f"practice:{record_id}",
            "source": "practice",
            "raw_uid": raw_uid,
            "raw_cid": raw_cid,
            "raw_qid": raw_qid,
            "raw_chapter_id": bank_row["raw_chapter_id"],
            "raw_set_id": bank_row["raw_set_id"],
            "timestamp": timestamp,
            "correct": 1 if raw_result == "1" else 0,
        }

    if not include_exam:
        return

    for fields in iter_mysql_insert_rows(sql_path, "pingshifen_exam_record"):
        record_id = normalize_sql_scalar(fields[0])
        raw_uid = normalize_sql_scalar(fields[1])
        raw_mid = normalize_sql_scalar(fields[2])
        raw_qid = normalize_sql_scalar(fields[3])
        raw_result = normalize_sql_scalar(fields[6])
        raw_state = normalize_sql_scalar(fields[7])
        raw_timestamp = normalize_sql_scalar(fields[8])

        if raw_state != "1":
            continue
        if raw_uid not in student_rows:
            continue
        if raw_qid not in bank_rows or raw_result not in {"1", "2"}:
            continue
        exam_row = exam_rows.get(raw_mid)
        if exam_row is None or exam_row["state"] != "1":
            continue

        raw_cid = exam_row["cid"]
        if class_memberships and raw_cid and (raw_uid, raw_cid) not in class_memberships:
            continue
        try:
            timestamp = int(raw_timestamp)
        except ValueError:
            continue

        bank_row = bank_rows[raw_qid]
        yield {
            "record_key": f"exam:{record_id}",
            "source": "exam",
            "raw_uid": raw_uid,
            "raw_cid": raw_cid,
            "raw_qid": raw_qid,
            "raw_chapter_id": bank_row["raw_chapter_id"],
            "raw_set_id": bank_row["raw_set_id"],
            "timestamp": timestamp,
            "correct": 1 if raw_result == "1" else 0,
        }


def build_doudouyun_question_text_from_sql(bank_row, chapter_row, set_row):
    content = bank_row["content"]
    options = []
    for label in ("a", "b", "c", "d"):
        value = bank_row[f"option_{label}"]
        if value:
            options.append(f"{label.upper()}. {value}")

    parts = [f"题目：{content}"]
    if options:
        parts.append("选项：" + " ".join(options))
    if bank_row["answer"]:
        parts.append(f"标准答案编码：{bank_row['answer']}")
    if chapter_row and chapter_row.get("title"):
        parts.append(f"章节：{chapter_row['title']}")
    if set_row and set_row.get("title"):
        parts.append(f"题集：{set_row['title']}")
    if bank_row["question_type"]:
        parts.append(f"题型编码：{bank_row['question_type']}")
    return " ".join(parts)


def build_doudouyun_kc_text_from_sql(raw_chapter_id, chapter_row, set_row, question_texts):
    parts = [f"豆豆云知识点 {raw_chapter_id}"]
    if chapter_row and chapter_row.get("title"):
        parts.append(f"章节名：{chapter_row['title']}")
    if set_row and set_row.get("title"):
        parts.append(f"题集：{set_row['title']}")
    preview = "；".join(question_texts[:3])
    if preview:
        parts.append(f"代表题目：{preview}")
    return "。".join(parts)


def _iter_doudouyun_raw_csvs(raw_root, stem):
    base_2024 = os.path.join(raw_root, "doudouyun", "2024")
    nested = []
    for child in sorted(os.listdir(base_2024)):
        child_path = os.path.join(base_2024, child)
        if os.path.isdir(child_path):
            candidate = os.path.join(child_path, f"{stem}_{child}.csv")
            if os.path.exists(candidate):
                nested.append(candidate)

    appoint = os.path.join(base_2024, f"appoint_{stem}.csv")
    if os.path.exists(appoint):
        nested.append(appoint)
    return nested


def load_doudouyun_2024_bank(raw_root):
    bank_rows = {}
    for path in _iter_doudouyun_raw_csvs(raw_root, "question_bank"):
        with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_qid = normalize_text(row.get("id"))
                if not raw_qid or raw_qid in bank_rows:
                    continue
                bank_rows[raw_qid] = {
                    "raw_qid": raw_qid,
                    "raw_set_id": normalize_text(row.get("set_id")),
                    "raw_chapter_id": normalize_text(row.get("chapter_id")),
                    "question_type": normalize_text(row.get("type")),
                    "content": normalize_text(row.get("content")),
                    "media_id": normalize_text(row.get("media_id")),
                    "option_a": normalize_text(row.get("option_a")),
                    "option_b": normalize_text(row.get("option_b")),
                    "option_c": normalize_text(row.get("option_c")),
                    "option_d": normalize_text(row.get("option_d")),
                    "answer": normalize_text(row.get("answer")),
                    "analysis": normalize_text(row.get("analysis")),
                    "status": normalize_text(row.get("status")),
                    "gmt_create": normalize_text(row.get("gmt_create")),
                    "gmt_modified": normalize_text(row.get("gmt_modified")),
                }
    if not bank_rows:
        raise FileNotFoundError("未找到 doudouyun 2024 question_bank CSV")
    return bank_rows


def load_doudouyun_2024_records(raw_root, bank_rows):
    records = []
    skipped = defaultdict(int)
    seen_record_ids = set()

    for path in _iter_doudouyun_raw_csvs(raw_root, "question_record"):
        with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                record_id = normalize_text(row.get("id"))
                if not record_id:
                    skipped["missing_record_id"] += 1
                    continue
                if record_id in seen_record_ids:
                    skipped["duplicate_record_id"] += 1
                    continue
                seen_record_ids.add(record_id)

                raw_qid = normalize_text(row.get("qid"))
                raw_uid = normalize_text(row.get("uid"))
                raw_result = normalize_text(row.get("result"))
                raw_timestamp = normalize_text(row.get("gmt_create"))

                if not raw_uid or not raw_qid or not raw_result or not raw_timestamp:
                    skipped["missing_core_fields"] += 1
                    continue
                if raw_qid not in bank_rows:
                    skipped["qid_missing_in_bank"] += 1
                    continue
                if raw_result not in {"1", "2"}:
                    skipped["unsupported_result"] += 1
                    continue

                bank_row = bank_rows[raw_qid]
                raw_chapter_id = normalize_text(row.get("chapter_id")) or bank_row["raw_chapter_id"]
                raw_set_id = normalize_text(row.get("set_id")) or bank_row["raw_set_id"]
                if not raw_chapter_id:
                    skipped["missing_chapter_id"] += 1
                    continue

                try:
                    timestamp = int(raw_timestamp)
                except ValueError:
                    skipped["bad_timestamp"] += 1
                    continue

                records.append(
                    {
                        "record_id": record_id,
                        "raw_uid": raw_uid,
                        "raw_qid": raw_qid,
                        "raw_chapter_id": raw_chapter_id,
                        "raw_set_id": raw_set_id,
                        "timestamp": timestamp,
                        "correct": 1 if raw_result == "1" else 0,
                    }
                )

    if not records:
        raise ValueError("未能从 doudouyun 2024 记录表中提取任何有效交互")

    return records, dict(skipped)


def split_contiguous_sequence(values, max_seq_len, min_seq_len):
    if len(values) < min_seq_len:
        return []
    if len(values) <= max_seq_len:
        return [values]

    n_chunks = ceil(len(values) / max_seq_len)
    base = len(values) // n_chunks
    remainder = len(values) % n_chunks
    chunk_sizes = [base + (1 if idx < remainder else 0) for idx in range(n_chunks)]

    chunks = []
    cursor = 0
    for size in chunk_sizes:
        chunk = values[cursor : cursor + size]
        cursor += size
        if len(chunk) >= min_seq_len:
            chunks.append(chunk)
    return chunks


def write_kt_sequence_file(path, sequences):
    with open(path, "w", encoding="utf-8") as f:
        for sample in sequences:
            q_values = ",".join(str(v) for v in sample["q"])
            s_values = ",".join(str(v) for v in sample["s"])
            f.write(f"{len(sample['q'])}\n")
            f.write(f"{q_values}\n")
            f.write(f"{s_values}\n")


def build_doudouyun_question_text(bank_row, raw_chapter_id, raw_set_id):
    content = bank_row["content"]
    options = []
    for label in ("a", "b", "c", "d"):
        value = bank_row[f"option_{label}"]
        if value:
            options.append(f"{label.upper()}. {value}")

    parts = [f"题目：{content}"]
    if options:
        parts.append("选项：" + " ".join(options))
    if bank_row["answer"]:
        parts.append(f"标准答案编码：{bank_row['answer']}")
    if bank_row["analysis"]:
        parts.append(f"解析：{bank_row['analysis']}")
    if raw_chapter_id:
        parts.append(f"章节ID：{raw_chapter_id}")
    if raw_set_id:
        parts.append(f"题集ID：{raw_set_id}")
    if bank_row["question_type"]:
        parts.append(f"题型编码：{bank_row['question_type']}")
    return " ".join(parts)


def build_doudouyun_kc_text(raw_chapter_id, question_texts, set_ids):
    preview = "；".join(question_texts[:3])
    parts = [f"豆豆云章节 {raw_chapter_id}"]
    if set_ids:
        parts.append("覆盖题集：" + "、".join(sorted(set_ids, key=lambda x: int(x))))
    if preview:
        parts.append(f"代表题目：{preview}")
    return "。".join(parts)


def rebuild_doudouyun_from_csv(
    raw_root,
    data_dir,
    text_output_dir,
    train_ratio=0.8,
    valid_ratio=0.1,
    test_ratio=0.1,
    seed=42,
    min_seq_len=3,
    max_seq_len=200,
):
    if abs((train_ratio + valid_ratio + test_ratio) - 1.0) > 1e-8:
        raise ValueError("train/valid/test 划分比例之和必须为 1.0")

    bank_rows = load_doudouyun_2024_bank(raw_root)
    records, skipped = load_doudouyun_2024_records(raw_root, bank_rows)

    records_by_user = defaultdict(list)
    observed_raw_qids = set()
    observed_raw_chapters = set()
    for record in records:
        records_by_user[record["raw_uid"]].append(record)
        observed_raw_qids.add(record["raw_qid"])
        observed_raw_chapters.add(record["raw_chapter_id"])

    eligible_users = []
    filtered_short_users = 0
    filtered_short_interactions = 0
    for raw_uid, user_records in records_by_user.items():
        user_records.sort(key=lambda item: (item["timestamp"], int(item["record_id"])))
        if len(user_records) < min_seq_len:
            filtered_short_users += 1
            filtered_short_interactions += len(user_records)
            continue
        eligible_users.append(raw_uid)

    eligible_users.sort(key=int)
    rng = np.random.default_rng(seed)
    rng.shuffle(eligible_users)

    n_users = len(eligible_users)
    if n_users < 3:
        raise ValueError(f"可用用户过少，无法切分 train/valid/test: {n_users}")

    train_user_count = max(1, int(round(n_users * train_ratio)))
    valid_user_count = max(1, int(round(n_users * valid_ratio)))
    if train_user_count + valid_user_count >= n_users:
        valid_user_count = max(1, min(valid_user_count, n_users - 2))
        train_user_count = max(1, n_users - valid_user_count - 1)

    train_users = set(eligible_users[:train_user_count])
    valid_users = set(eligible_users[train_user_count : train_user_count + valid_user_count])
    test_users = set(eligible_users[train_user_count + valid_user_count :])

    observed_raw_qids = sorted(observed_raw_qids, key=int)
    observed_raw_chapters = sorted(observed_raw_chapters, key=int)
    raw_qid_to_dense = {
        raw_qid: dense_qid for dense_qid, raw_qid in enumerate(observed_raw_qids, start=1)
    }
    raw_chapter_to_kc = {
        raw_chapter: kc_id for kc_id, raw_chapter in enumerate(observed_raw_chapters)
    }

    split_sequences = {"train": [], "valid": [], "test": []}
    split_stats = {
        "train": {"users": 0, "interactions": 0, "segments": 0},
        "valid": {"users": 0, "interactions": 0, "segments": 0},
        "test": {"users": 0, "interactions": 0, "segments": 0},
    }

    chapter_question_texts = defaultdict(list)
    chapter_set_ids = defaultdict(set)
    question_texts = {}

    for raw_qid in observed_raw_qids:
        bank_row = bank_rows[raw_qid]
        raw_chapter_id = bank_row["raw_chapter_id"]
        raw_set_id = bank_row["raw_set_id"]
        dense_qid = raw_qid_to_dense[raw_qid]
        kc_id = raw_chapter_to_kc[raw_chapter_id]
        question_text = build_doudouyun_question_text(bank_row, raw_chapter_id, raw_set_id)
        question_texts[str(dense_qid)] = {
            "text": question_text,
            "content": question_text,
            "skill": kc_id,
            "raw_qid": raw_qid,
            "raw_chapter_id": raw_chapter_id,
            "raw_set_id": raw_set_id,
            "question_type": bank_row["question_type"],
            "answer": bank_row["answer"],
            "source": "doudouyun_2024",
        }
        chapter_question_texts[raw_chapter_id].append(bank_row["content"])
        if raw_set_id:
            chapter_set_ids[raw_chapter_id].add(raw_set_id)

    kc_texts = {}
    for raw_chapter_id in observed_raw_chapters:
        kc_id = raw_chapter_to_kc[raw_chapter_id]
        kc_texts[str(kc_id)] = {
            "text": build_doudouyun_kc_text(
                raw_chapter_id,
                chapter_question_texts[raw_chapter_id],
                chapter_set_ids[raw_chapter_id],
            ),
            "raw_chapter_id": raw_chapter_id,
            "source": "doudouyun_2024",
        }

    for raw_uid in eligible_users:
        user_records = records_by_user[raw_uid]
        q_sequence = [raw_qid_to_dense[item["raw_qid"]] for item in user_records]
        s_sequence = [item["correct"] for item in user_records]
        chunks = list(
            zip(
                split_contiguous_sequence(q_sequence, max_seq_len=max_seq_len, min_seq_len=min_seq_len),
                split_contiguous_sequence(s_sequence, max_seq_len=max_seq_len, min_seq_len=min_seq_len),
            )
        )
        if raw_uid in train_users:
            split_name = "train"
        elif raw_uid in valid_users:
            split_name = "valid"
        else:
            split_name = "test"

        split_stats[split_name]["users"] += 1
        split_stats[split_name]["interactions"] += len(user_records)
        split_stats[split_name]["segments"] += len(chunks)
        for q_chunk, s_chunk in chunks:
            split_sequences[split_name].append(
                {
                    "q": q_chunk,
                    "s": s_chunk,
                    "raw_uid": raw_uid,
                }
            )

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(text_output_dir, exist_ok=True)

    write_kt_sequence_file(os.path.join(data_dir, "train.txt"), split_sequences["train"])
    write_kt_sequence_file(os.path.join(data_dir, "valid.txt"), split_sequences["valid"])
    write_kt_sequence_file(os.path.join(data_dir, "test.txt"), split_sequences["test"])

    question_file = os.path.join(text_output_dir, "doudouyun_question_texts.json")
    kc_file = os.path.join(text_output_dir, "doudouyun_kc_texts.json")
    with open(question_file, "w", encoding="utf-8") as f:
        json.dump(question_texts, f, ensure_ascii=False, indent=2)
    with open(kc_file, "w", encoding="utf-8") as f:
        json.dump(kc_texts, f, ensure_ascii=False, indent=2)

    summary = {
        "dataset": "doudouyun",
        "source": "doudouyun_2024",
        "raw_record_count": len(records),
        "raw_bank_question_count": len(bank_rows),
        "observed_question_count": len(observed_raw_qids),
        "observed_kc_count": len(observed_raw_chapters),
        "eligible_user_count": len(eligible_users),
        "filtered_short_users": filtered_short_users,
        "filtered_short_interactions": filtered_short_interactions,
        "skipped_records": skipped,
        "split_stats": split_stats,
        "train_ratio": train_ratio,
        "valid_ratio": valid_ratio,
        "test_ratio": test_ratio,
        "seed": seed,
        "min_seq_len": min_seq_len,
        "max_seq_len": max_seq_len,
    }

    print("✅ 已根据 doudouyun 2024 原始 CSV 重建数据集")
    print(
        f"   观测题目数={summary['observed_question_count']}, "
        f"知识点数={summary['observed_kc_count']}, 用户数={summary['eligible_user_count']}"
    )
    for split_name in ("train", "valid", "test"):
        stats = split_stats[split_name]
        print(
            f"   {split_name}: users={stats['users']}, interactions={stats['interactions']}, "
            f"segments={stats['segments']}"
        )
    if filtered_short_users or skipped:
        print(
            f"   过滤短序列用户={filtered_short_users}, "
            f"跳过记录={sum(skipped.values())}"
        )

    return summary


def write_user_record_to_bucket(handle, record):
    handle.write(
        "\t".join(
            [
                record["raw_uid"],
                str(record["timestamp"]),
                record["record_key"],
                record["raw_qid"],
                str(record["correct"]),
            ]
        )
        + "\n"
    )


def rebuild_doudouyun_from_sql(
    sql_path,
    data_dir,
    text_output_dir,
    train_ratio=0.8,
    valid_ratio=0.1,
    test_ratio=0.1,
    seed=42,
    min_seq_len=3,
    max_seq_len=200,
    include_exam=True,
    n_buckets=128,
):
    if abs((train_ratio + valid_ratio + test_ratio) - 1.0) > 1e-8:
        raise ValueError("train/valid/test 划分比例之和必须为 1.0")

    dimensions = load_doudouyun_sql_dimensions(sql_path)
    bank_rows = dimensions["bank"]
    chapter_rows = dimensions["chapters"]
    set_rows = dimensions["sets"]

    user_counts = defaultdict(int)
    observed_raw_qids = set()
    observed_raw_chapters = set()
    source_counts = defaultdict(int)
    total_records = 0

    tmp_dir = tempfile.mkdtemp(prefix="trisg_doudouyun_")
    bucket_paths = [os.path.join(tmp_dir, f"bucket_{idx:03d}.tsv") for idx in range(n_buckets)]
    bucket_handles = [open(path, "w", encoding="utf-8") for path in bucket_paths]

    try:
        try:
            for record in iter_doudouyun_sql_records(sql_path, dimensions, include_exam=include_exam):
                bucket_idx = hash(record["raw_uid"]) % n_buckets
                write_user_record_to_bucket(bucket_handles[bucket_idx], record)
                user_counts[record["raw_uid"]] += 1
                observed_raw_qids.add(record["raw_qid"])
                observed_raw_chapters.add(record["raw_chapter_id"])
                source_counts[record["source"]] += 1
                total_records += 1
        finally:
            for handle in bucket_handles:
                handle.close()

        eligible_users = sorted(
            [raw_uid for raw_uid, count in user_counts.items() if count >= min_seq_len],
            key=int,
        )
        filtered_short_users = sum(1 for count in user_counts.values() if count < min_seq_len)
        filtered_short_interactions = sum(
            count for count in user_counts.values() if count < min_seq_len
        )

        rng = np.random.default_rng(seed)
        rng.shuffle(eligible_users)
        n_users = len(eligible_users)
        if n_users < 3:
            raise ValueError(f"可用用户过少，无法切分 train/valid/test: {n_users}")

        train_user_count = max(1, int(round(n_users * train_ratio)))
        valid_user_count = max(1, int(round(n_users * valid_ratio)))
        if train_user_count + valid_user_count >= n_users:
            valid_user_count = max(1, min(valid_user_count, n_users - 2))
            train_user_count = max(1, n_users - valid_user_count - 1)

        train_users = set(eligible_users[:train_user_count])
        valid_users = set(eligible_users[train_user_count : train_user_count + valid_user_count])
        test_users = set(eligible_users[train_user_count + valid_user_count :])

        eligible_raw_qids = set()
        eligible_raw_chapters = set()
        for bucket_path in bucket_paths:
            with open(bucket_path, "r", encoding="utf-8") as f:
                for line in f:
                    raw_uid, _, _, raw_qid, _ = line.rstrip("\n").split("\t")
                    if raw_uid not in train_users and raw_uid not in valid_users and raw_uid not in test_users:
                        continue
                    eligible_raw_qids.add(raw_qid)
                    eligible_raw_chapters.add(bank_rows[raw_qid]["raw_chapter_id"])

        observed_raw_qids = sorted(eligible_raw_qids, key=int)
        observed_raw_chapters = sorted(eligible_raw_chapters, key=int)
        raw_qid_to_dense = {
            raw_qid: dense_qid for dense_qid, raw_qid in enumerate(observed_raw_qids, start=1)
        }
        raw_chapter_to_kc = {
            raw_chapter: kc_id for kc_id, raw_chapter in enumerate(observed_raw_chapters)
        }

        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(text_output_dir, exist_ok=True)

        split_files = {
            "train": open(os.path.join(data_dir, "train.txt"), "w", encoding="utf-8"),
            "valid": open(os.path.join(data_dir, "valid.txt"), "w", encoding="utf-8"),
            "test": open(os.path.join(data_dir, "test.txt"), "w", encoding="utf-8"),
        }
        split_stats = {
            "train": {"users": 0, "interactions": 0, "segments": 0},
            "valid": {"users": 0, "interactions": 0, "segments": 0},
            "test": {"users": 0, "interactions": 0, "segments": 0},
        }

        try:
            for bucket_path in bucket_paths:
                bucket_users = defaultdict(list)
                with open(bucket_path, "r", encoding="utf-8") as f:
                    for line in f:
                        raw_uid, timestamp, record_key, raw_qid, correct = line.rstrip("\n").split("\t")
                        if raw_uid not in train_users and raw_uid not in valid_users and raw_uid not in test_users:
                            continue
                        bucket_users[raw_uid].append(
                            (int(timestamp), record_key, raw_qid, int(correct))
                        )

                for raw_uid, records in bucket_users.items():
                    records.sort(key=lambda item: (item[0], item[1]))
                    q_sequence = [raw_qid_to_dense[item[2]] for item in records]
                    s_sequence = [item[3] for item in records]
                    q_chunks = split_contiguous_sequence(
                        q_sequence,
                        max_seq_len=max_seq_len,
                        min_seq_len=min_seq_len,
                    )
                    s_chunks = split_contiguous_sequence(
                        s_sequence,
                        max_seq_len=max_seq_len,
                        min_seq_len=min_seq_len,
                    )

                    if raw_uid in train_users:
                        split_name = "train"
                    elif raw_uid in valid_users:
                        split_name = "valid"
                    else:
                        split_name = "test"

                    split_stats[split_name]["users"] += 1
                    split_stats[split_name]["interactions"] += len(records)
                    split_stats[split_name]["segments"] += len(q_chunks)

                    handle = split_files[split_name]
                    for q_chunk, s_chunk in zip(q_chunks, s_chunks):
                        handle.write(f"{len(q_chunk)}\n")
                        handle.write(",".join(str(v) for v in q_chunk) + "\n")
                        handle.write(",".join(str(v) for v in s_chunk) + "\n")
        finally:
            for handle in split_files.values():
                handle.close()

        question_texts = {}
        chapter_question_texts = defaultdict(list)
        for raw_qid in observed_raw_qids:
            bank_row = bank_rows[raw_qid]
            raw_chapter_id = bank_row["raw_chapter_id"]
            raw_set_id = bank_row["raw_set_id"]
            chapter_row = chapter_rows.get(raw_chapter_id)
            set_row = set_rows.get(raw_set_id)
            dense_qid = raw_qid_to_dense[raw_qid]
            kc_id = raw_chapter_to_kc[raw_chapter_id]
            question_text = build_doudouyun_question_text_from_sql(
                bank_row,
                chapter_row,
                set_row,
            )
            question_texts[str(dense_qid)] = {
                "text": question_text,
                "content": question_text,
                "skill": kc_id,
                "raw_qid": raw_qid,
                "raw_chapter_id": raw_chapter_id,
                "raw_set_id": raw_set_id,
                "chapter_title": chapter_row.get("title") if chapter_row else "",
                "set_title": set_row.get("title") if set_row else "",
                "question_type": bank_row["question_type"],
                "answer": bank_row["answer"],
                "source": "doudouyun_sql_20240928",
            }
            chapter_question_texts[raw_chapter_id].append(bank_row["content"])

        kc_texts = {}
        for raw_chapter_id in observed_raw_chapters:
            chapter_row = chapter_rows.get(raw_chapter_id)
            set_row = set_rows.get(chapter_row["raw_set_id"]) if chapter_row else None
            kc_id = raw_chapter_to_kc[raw_chapter_id]
            kc_texts[str(kc_id)] = {
                "text": build_doudouyun_kc_text_from_sql(
                    raw_chapter_id,
                    chapter_row,
                    set_row,
                    chapter_question_texts[raw_chapter_id],
                ),
                "raw_chapter_id": raw_chapter_id,
                "chapter_title": chapter_row.get("title") if chapter_row else "",
                "raw_set_id": chapter_row.get("raw_set_id") if chapter_row else "",
                "set_title": set_row.get("title") if set_row else "",
                "source": "doudouyun_sql_20240928",
            }

        question_file = os.path.join(text_output_dir, "doudouyun_question_texts.json")
        kc_file = os.path.join(text_output_dir, "doudouyun_kc_texts.json")
        with open(question_file, "w", encoding="utf-8") as f:
            json.dump(question_texts, f, ensure_ascii=False, indent=2)
        with open(kc_file, "w", encoding="utf-8") as f:
            json.dump(kc_texts, f, ensure_ascii=False, indent=2)

        summary = {
            "dataset": "doudouyun",
            "source": "doudouyun_sql_20240928",
            "sql_path": sql_path,
            "include_exam": include_exam,
            "raw_record_count": total_records,
            "raw_bank_question_count": len(bank_rows),
            "observed_question_count": len(observed_raw_qids),
            "observed_kc_count": len(observed_raw_chapters),
            "eligible_user_count": len(eligible_users),
            "filtered_short_users": filtered_short_users,
            "filtered_short_interactions": filtered_short_interactions,
            "source_counts": dict(source_counts),
            "split_stats": split_stats,
            "train_ratio": train_ratio,
            "valid_ratio": valid_ratio,
            "test_ratio": test_ratio,
            "seed": seed,
            "min_seq_len": min_seq_len,
            "max_seq_len": max_seq_len,
        }

        print("✅ 已根据 doudouyun SQL 全量重建数据集")
        print(
            f"   观测题目数={summary['observed_question_count']}, "
            f"知识点数={summary['observed_kc_count']}, 用户数={summary['eligible_user_count']}"
        )
        print(
            f"   练习记录={source_counts.get('practice', 0)}, "
            f"考试记录={source_counts.get('exam', 0)}"
        )
        for split_name in ("train", "valid", "test"):
            stats = split_stats[split_name]
            print(
                f"   {split_name}: users={stats['users']}, interactions={stats['interactions']}, "
                f"segments={stats['segments']}"
            )
        if filtered_short_users:
            print(
                f"   过滤短序列用户={filtered_short_users}, "
                f"过滤交互={filtered_short_interactions}"
            )

        return summary
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def rebuild_doudouyun_from_raw(
    raw_root,
    data_dir,
    text_output_dir,
    train_ratio=0.8,
    valid_ratio=0.1,
    test_ratio=0.1,
    seed=42,
    min_seq_len=3,
    max_seq_len=200,
    include_exam=True,
):
    sql_paths = build_doudouyun_sql_paths(raw_root)
    if os.path.exists(sql_paths["latest"]):
        return rebuild_doudouyun_from_sql(
            sql_paths["latest"],
            data_dir=data_dir,
            text_output_dir=text_output_dir,
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            test_ratio=test_ratio,
            seed=seed,
            min_seq_len=min_seq_len,
            max_seq_len=max_seq_len,
            include_exam=include_exam,
        )
    return rebuild_doudouyun_from_csv(
        raw_root=raw_root,
        data_dir=data_dir,
        text_output_dir=text_output_dir,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
        seed=seed,
        min_seq_len=min_seq_len,
        max_seq_len=max_seq_len,
    )
