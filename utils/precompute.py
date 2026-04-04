import json
import os
import pickle
import sys
import types
from enum import Enum
from importlib.machinery import ModuleSpec

import torch
from tqdm import tqdm

from utils.project import project_path


def _install_torchvision_stub():
    """为当前环境提供最小 torchvision stub，避免 transformers 在文本链路中误触发损坏的 vision 依赖。"""
    if "torchvision" in sys.modules:
        return

    class InterpolationMode(Enum):
        NEAREST = "nearest"
        NEAREST_EXACT = "nearest-exact"
        BILINEAR = "bilinear"
        BICUBIC = "bicubic"
        BOX = "box"
        HAMMING = "hamming"
        LANCZOS = "lanczos"

    torchvision_module = types.ModuleType("torchvision")
    torchvision_module.__spec__ = ModuleSpec("torchvision", loader=None)

    transforms_module = types.ModuleType("torchvision.transforms")
    transforms_module.__spec__ = ModuleSpec("torchvision.transforms", loader=None)
    transforms_module.InterpolationMode = InterpolationMode
    torchvision_module.transforms = transforms_module

    for submodule_name in ("io", "datasets", "models", "ops", "utils", "_meta_registrations"):
        module_name = f"torchvision.{submodule_name}"
        submodule = types.ModuleType(module_name)
        submodule.__spec__ = ModuleSpec(module_name, loader=None)
        setattr(torchvision_module, submodule_name, submodule)
        sys.modules[module_name] = submodule

    sys.modules["torchvision"] = torchvision_module
    sys.modules["torchvision.transforms"] = transforms_module


_install_torchvision_stub()

from transformers import AutoModel, AutoTokenizer


class QwenEmbeddingGenerator:
    """Qwen 语义嵌入生成器。"""

    def __init__(self, model_path="pretrained_models/qwen3-4b", device="cuda"):
        self.device = device if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
        self.model_path = model_path
        print(f"📦 加载 Qwen 模型: {model_path}")
        print(f"🔧 设备: {self.device}")

        local_only = os.path.isdir(model_path)
        common_kwargs = {"trust_remote_code": True}
        if local_only:
            common_kwargs["local_files_only"] = True

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False,
            **common_kwargs,
        )
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        torch_dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self.model = AutoModel.from_pretrained(
            model_path,
            dtype=torch_dtype,
            **common_kwargs,
        )
        self.model.to(self.device)
        self.model.eval()
        self.hidden_size = int(getattr(self.model.config, "hidden_size"))
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

    @staticmethod
    def is_cuda_oom(exc):
        message = str(exc).lower()
        return "out of memory" in message or "cuda out of memory" in message

    def _clear_cuda_cache(self):
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def batch_encode_texts(self, texts, batch_size=32, max_length=256):
        embeddings = []

        current_batch_size = max(1, int(batch_size))
        current_max_length = max(64, int(max_length))
        progress = tqdm(total=len(texts), desc="编码", leave=False)
        start = 0

        try:
            with torch.no_grad():
                while start < len(texts):
                    batch_texts = texts[start : start + current_batch_size]
                    try:
                        encoded = self.tokenizer(
                            batch_texts,
                            padding=True,
                            truncation=True,
                            max_length=current_max_length,
                            return_tensors="pt",
                        )
                        encoded = {key: value.to(self.device) for key, value in encoded.items()}
                        outputs = self.model(**encoded)
                        hidden_states = outputs.last_hidden_state
                        attention_mask = encoded["attention_mask"].unsqueeze(-1)
                        masked_hidden = hidden_states * attention_mask
                        pooled = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1)
                        embeddings.append(pooled.detach().float().cpu())
                        start += len(batch_texts)
                        progress.update(len(batch_texts))
                    except torch.OutOfMemoryError as exc:
                        if not self.device.startswith("cuda") or not self.is_cuda_oom(exc):
                            raise
                        self._clear_cuda_cache()
                        if current_batch_size > 1:
                            next_batch_size = max(1, current_batch_size // 2)
                            print(
                                f"⚠️  预计算发生 CUDA OOM，自动将 batch_size "
                                f"{current_batch_size} -> {next_batch_size}"
                            )
                            current_batch_size = next_batch_size
                            continue
                        if current_max_length > 128:
                            next_max_length = max(128, current_max_length // 2)
                            print(
                                f"⚠️  单条样本仍触发 CUDA OOM，自动将 max_length "
                                f"{current_max_length} -> {next_max_length}"
                            )
                            current_max_length = next_max_length
                            continue
                        raise
        finally:
            progress.close()

        return torch.cat(embeddings, dim=0).numpy()

    def precompute_question_embeddings(
        self,
        questions_data,
        output_path,
        batch_size=32,
        dataset_name=None,
        max_length=256,
    ):
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
            max_length=max_length,
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

    def precompute_kc_embeddings(
        self,
        kc_data,
        output_path,
        batch_size=32,
        dataset_name=None,
        max_length=256,
    ):
        print(f"📚 预计算 {len(kc_data)} 个知识点嵌入...")

        kc_ids = sorted(kc_data.keys(), key=lambda x: int(x))
        kc_texts = [self.normalize_text_entry(kc_data[kid]) for kid in kc_ids]

        kc_embeddings = self.batch_encode_texts(
            kc_texts,
            batch_size=batch_size,
            max_length=max_length,
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
