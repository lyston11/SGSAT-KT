import copy

import tomlkit
import yaml

from utils.project import CONFIG_DIR, DATA_DIR


def merge_dicts(base, update):
    """递归合并字典。"""
    result = copy.deepcopy(base)
    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def flatten_config(config, parent_key="", sep="_"):
    """将嵌套配置扁平化，同时保留兼容别名。"""
    items = []
    for key, value in config.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            items.extend(flatten_config(value, new_key, sep=sep).items())
        else:
            items.append((new_key, value))
        items.append((key, value))

    flat_dict = dict(items)
    if "training_n_epochs" in flat_dict:
        flat_dict["epochs"] = flat_dict["training_n_epochs"]
        flat_dict["n_epochs"] = flat_dict["training_n_epochs"]
    if "training_batch_size" in flat_dict:
        flat_dict["batch_size"] = flat_dict["training_batch_size"]
    if "training_learning_rate" in flat_dict:
        flat_dict["learning_rate"] = flat_dict["training_learning_rate"]
    if "training_device" in flat_dict:
        flat_dict["device"] = flat_dict["training_device"]
    if "llm_use_llm" in flat_dict:
        flat_dict["use_llm"] = flat_dict["llm_use_llm"]
    if "gnn_use_gnn" in flat_dict:
        flat_dict["use_gnn"] = flat_dict["gnn_use_gnn"]
    if "llm_pretrained_model" in flat_dict:
        flat_dict["pretrained_model"] = flat_dict["llm_pretrained_model"]
    if "precomputed_use_precomputed" in flat_dict:
        flat_dict["use_precomputed"] = flat_dict["precomputed_use_precomputed"]
    return flat_dict


def load_yaml_config(config_name="default.yaml"):
    config_path = CONFIG_DIR / config_name
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_mode_config(mode, dataset=None, device=None, config_name="default.yaml"):
    default_config = load_yaml_config(config_name)
    presets = default_config.get("presets", {})
    config = copy.deepcopy(default_config)
    config.pop("presets", None)
    if mode in presets:
        config = merge_dicts(config, presets[mode])
    if dataset:
        config.setdefault("training", {})["dataset"] = dataset
    if device:
        config.setdefault("training", {})["device"] = device
    return config


def load_dataset_registry():
    with (DATA_DIR / "datasets.toml").open("r", encoding="utf-8") as f:
        return tomlkit.load(f)
