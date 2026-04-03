import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from DTransformer.data import Batch, KTData, transform_batch


class KTDataSubset:
    def __init__(self, base_data, indices, batch_size, shuffle=False):
        self.base_data = base_data
        self.indices = list(indices)
        subset = Subset(base_data, self.indices)
        self.loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=transform_batch,
            num_workers=0,
        )
        self.inputs = base_data.inputs
        self.seq_len = base_data.seq_len

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return len(self.indices)


class FlexibleKTData:
    def __init__(self, data_path, inputs, batch_size=1, seq_len=None, shuffle=False):
        self.samples = parse_flexible_kt_samples(data_path, inputs)
        self.inputs = inputs
        self.seq_len = seq_len
        self.loader = DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=transform_batch,
            num_workers=0,
        )

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return Batch(
            torch.tensor(self.samples[index], dtype=torch.long),
            self.inputs,
            self.seq_len,
        )


def is_seq_len_header(line):
    stripped = line.strip()
    return stripped.isdigit() and "," not in stripped


def is_binary_like_line(line, expected_len=None):
    stripped = line.strip()
    if not stripped or "," not in stripped:
        return False
    parts = stripped.split(",")
    if expected_len is not None and len(parts) != expected_len:
        return False
    try:
        return all(part in {"0", "1", "-1"} for part in parts)
    except Exception:
        return False


def sniff_data_format(data_path, inputs, preview_groups=8):
    expected_group = len(inputs) + 1
    with open(data_path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    if not lines:
        return {"mode": "standard", "reason": "empty_file"}

    if len(lines) >= expected_group:
        first_group = lines[:expected_group]
        if is_seq_len_header(first_group[0]):
            return {"mode": "standard", "reason": "seq_len_header"}

    seq_counter = 0
    idx = 0
    while idx < len(lines) and seq_counter < preview_groups:
        header = lines[idx].strip()
        if not header:
            idx += 1
            continue
        try:
            seq_len = int(header)
        except ValueError:
            return {
                "mode": "flexible",
                "reason": f"group_{seq_counter}_header_not_seq_len",
            }

        start = idx + 1
        end = start + len(inputs)
        if end > len(lines):
            return {
                "mode": "flexible",
                "reason": f"group_{seq_counter}_incomplete_tail",
            }

        field_lines = [lines[pos].strip() for pos in range(start, end)]
        try:
            field_lengths = [len(line.split(",")) for line in field_lines]
        except Exception:
            return {
                "mode": "flexible",
                "reason": f"group_{seq_counter}_split_failed",
            }

        if any(length != seq_len for length in field_lengths):
            return {
                "mode": "flexible",
                "reason": (
                    f"group_{seq_counter}_field_len_mismatch_{field_lengths}_seq_{seq_len}"
                ),
            }

        idx = end
        seq_counter += 1

    return {"mode": "standard", "reason": "validated_preview"}


def parse_flexible_kt_samples(data_path, inputs):
    samples = []
    skipped = 0
    with open(data_path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    idx = 0
    expected_fields = len(inputs)
    while idx < len(lines):
        header = lines[idx].strip()
        if not header:
            idx += 1
            continue

        try:
            seq_len = int(header)
        except ValueError:
            idx += 1
            skipped += 1
            continue

        cursor = idx + 1
        field_lines = []
        while cursor < len(lines) and len(field_lines) < expected_fields:
            candidate = lines[cursor].strip()
            if candidate:
                field_lines.append(candidate)
            cursor += 1

        if len(field_lines) < expected_fields:
            skipped += 1
            break

        parsed_fields = []
        valid = True
        for field_line in field_lines:
            parts = field_line.split(",") if field_line else []
            try:
                values = [int(part) for part in parts]
            except ValueError:
                valid = False
                break
            if len(values) != seq_len:
                valid = False
                break
            parsed_fields.append(values)

        if valid:
            samples.append(parsed_fields)
            idx = cursor
            continue

        next_start = None
        for probe in range(idx + 1, len(lines)):
            if is_seq_len_header(lines[probe]):
                probe_fields = []
                probe_cursor = probe + 1
                while probe_cursor < len(lines) and len(probe_fields) < expected_fields:
                    candidate = lines[probe_cursor].strip()
                    if candidate:
                        probe_fields.append(candidate)
                    probe_cursor += 1
                if len(probe_fields) < expected_fields:
                    continue
                try:
                    probe_seq_len = int(lines[probe].strip())
                except ValueError:
                    continue
                if all(
                    len(field_line.split(",")) == probe_seq_len for field_line in probe_fields
                ):
                    next_start = probe
                    break

        skipped += 1
        idx = next_start if next_start is not None else cursor

    print(f"⚠️  使用兼容解析器加载 {data_path} 时跳过了 {skipped} 处异常片段")
    return samples


def build_data_source(data_path, inputs, batch_size, seq_len=None, shuffle=False):
    sniff = sniff_data_format(data_path, inputs)
    if sniff["mode"] == "flexible":
        print(
            f"⚠️  检测到非标准序列文件格式，启用兼容解析器: {data_path} "
            f"({sniff['reason']})"
        )
        return FlexibleKTData(
            data_path,
            inputs,
            batch_size=batch_size,
            seq_len=seq_len,
            shuffle=shuffle,
        )
    return KTData(
        data_path,
        inputs,
        batch_size=batch_size,
        seq_len=seq_len,
        shuffle=shuffle,
    )


def build_generated_valid_split(
    train_path,
    inputs,
    seq_len,
    train_batch_size,
    eval_batch_size,
    valid_ratio=0.1,
    seed=42,
):
    if not (0.0 < valid_ratio < 1.0):
        raise ValueError(f"validation_ratio 必须在 (0,1) 内，当前为 {valid_ratio}")

    base_data = build_data_source(
        train_path,
        inputs,
        batch_size=1,
        seq_len=seq_len,
        shuffle=False,
    )
    num_samples = len(base_data)
    if num_samples < 2:
        raise ValueError(f"训练数据样本数过少，无法划分验证集: {num_samples}")

    indices = np.arange(num_samples)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    valid_size = max(1, int(round(num_samples * valid_ratio)))
    valid_size = min(valid_size, num_samples - 1)
    valid_indices = indices[:valid_size]
    train_indices = indices[valid_size:]

    train_data = KTDataSubset(
        base_data,
        train_indices,
        batch_size=train_batch_size,
        shuffle=True,
    )
    valid_data = KTDataSubset(
        base_data,
        valid_indices,
        batch_size=eval_batch_size,
        shuffle=False,
    )
    split_info = {
        "source": "generated_from_train",
        "train_path": train_path,
        "valid_ratio": valid_ratio,
        "valid_seed": seed,
        "train_size": len(train_indices),
        "valid_size": len(valid_indices),
    }
    return train_data, valid_data, split_info
