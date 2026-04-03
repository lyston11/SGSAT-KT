import json
import os
import pickle

import torch
from sentence_transformers import SentenceTransformer

from utils.project import project_path


class QwenEmbeddingGenerator:
    """Qwen 语义嵌入生成器 (使用 sentence-transformers)"""

    def __init__(self, model_path="pretrained_models/qwen3-4b", device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_path = model_path
        print(f"📦 加载 Qwen 模型: {model_path}")
        print(f"🔧 设备: {self.device}")

        local_only = os.path.isdir(model_path)
        model_kwargs = {"trust_remote_code": True}
        if local_only:
            model_kwargs["local_files_only"] = True

        self.model = SentenceTransformer(
            model_path,
            device=self.device,
            model_kwargs=model_kwargs,
            tokenizer_kwargs={"local_files_only": local_only},
        )
        self.hidden_size = self.model.get_sentence_embedding_dimension()
        print(f"✅ 模型加载完成，嵌入维度: {self.hidden_size}")

    @staticmethod
    def normalize_text_entry(entry):
        if isinstance(entry, dict):
            for key in ("text", "content", "description", "name"):
                value = entry.get(key, "")
                if value:
                    return str(value)
            return json.dumps(entry, ensure_ascii=False)
        if entry is None:
            return ""
        return str(entry)

    def batch_encode_texts(self, texts, batch_size=32):
        return self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

    def precompute_question_embeddings(self, questions_data, output_path, batch_size=32, dataset_name=None):
        print(f"📝 预计算 {len(questions_data)} 个题目嵌入...")

        question_ids = sorted(questions_data.keys(), key=lambda x: int(x))
        question_texts = [
            self.normalize_text_entry(questions_data[qid])
            if not isinstance(questions_data[qid], dict)
            else self.normalize_text_entry(
                questions_data[qid].get("text") or questions_data[qid].get("content", "")
            )
            for qid in question_ids
        ]

        question_embeddings = self.batch_encode_texts(
            question_texts,
            batch_size=batch_size,
        )

        result = {
            "question_ids": question_ids,
            "embeddings": question_embeddings,
            "hidden_size": self.hidden_size,
            "model_path": self.model_path,
            "dataset_name": dataset_name,
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(result, f)

        print(f"✅ 题目嵌入已保存: {output_path}")
        return result

    def precompute_kc_embeddings(self, kc_data, output_path, batch_size=32, dataset_name=None):
        print(f"📚 预计算 {len(kc_data)} 个知识点嵌入...")

        kc_ids = sorted(kc_data.keys(), key=lambda x: int(x))
        kc_texts = [self.normalize_text_entry(kc_data[kid]) for kid in kc_ids]

        kc_embeddings = self.batch_encode_texts(
            kc_texts,
            batch_size=batch_size,
        )

        result = {
            "kc_ids": kc_ids,
            "embeddings": kc_embeddings,
            "hidden_size": self.hidden_size,
            "model_path": self.model_path,
            "dataset_name": dataset_name,
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(result, f)

        print(f"✅ 知识点嵌入已保存: {output_path}")
        return result


def resolve_precompute_model_path(config):
    model_path_cfg = (
        config.get("llm", {}).get("pretrained_model")
        or "pretrained_models/qwen3-4b"
    )

    if os.path.isabs(model_path_cfg):
        resolved_model_path = model_path_cfg
    else:
        resolved_model_path = project_path(model_path_cfg)

    legacy_model_path = project_path("pretrained_models", "Qwen3-Embedding-4B")
    if not os.path.isdir(resolved_model_path) and os.path.isdir(legacy_model_path):
        print(f"⚠️  配置路径不存在，回退使用旧目录: {legacy_model_path}")
        resolved_model_path = legacy_model_path

    if not os.path.isdir(resolved_model_path):
        raise FileNotFoundError(
            f"本地模型目录不存在: {resolved_model_path}\n"
            f"请检查 configs/default.yaml 的 llm.pretrained_model，"
            f"当前建议为 pretrained_models/qwen3-4b"
        )

    return resolved_model_path


def load_precompute_text_assets(dataset_name, text_data_dir):
    q_file = os.path.join(text_data_dir, f"{dataset_name}_question_texts.json")
    kc_file = os.path.join(text_data_dir, f"{dataset_name}_kc_texts.json")

    questions = None
    kcs = None
    if os.path.exists(q_file):
        with open(q_file, "r", encoding="utf-8") as f:
            questions = json.load(f)
    if os.path.exists(kc_file):
        with open(kc_file, "r", encoding="utf-8") as f:
            kcs = json.load(f)

    return {
        "question_path": q_file,
        "kc_path": kc_file,
        "questions": questions,
        "kcs": kcs,
    }


def backfill_missing_kc_texts(kcs, questions):
    if questions is None or kcs is None:
        return kcs, []

    kc_to_questions = {}
    for _, info in questions.items():
        skill = info.get("skill", info.get("kc", None))
        if skill is not None and int(skill) >= 0:
            kc_to_questions.setdefault(int(skill), []).append(
                info.get("text", info.get("content", ""))
            )

    missing_kcs = []
    for kc_id, q_texts_list in sorted(kc_to_questions.items()):
        kc_id_str = str(kc_id)
        if kc_id_str not in kcs:
            kcs[kc_id_str] = "；".join(q_texts_list[:3])
            missing_kcs.append(kc_id)

    return kcs, missing_kcs
