"""
统一配置加载器
支持多种配置格式和配置继承
"""
import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigLoader:
    """统一配置加载器"""

    def __init__(self, config_dir: str = None):
        if config_dir is None:
            # 默认配置目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(current_dir, "../../"))
            config_dir = os.path.join(project_root, "configs")

        self.config_dir = Path(config_dir)
        self.base_config = {}
        self.user_config = {}
        self.merged_config = {}

    def load_yaml(self, config_file: str) -> Dict[str, Any]:
        """加载YAML配置文件"""
        config_path = self.config_dir / config_file
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        return config

    def load_json(self, config_file: str) -> Dict[str, Any]:
        """加载JSON配置文件"""
        config_path = self.config_dir / config_file
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        return config

    def merge_configs(self, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """合并多个配置（后面的覆盖前面的）"""
        merged = {}
        for config in configs:
            merged = self._deep_update(merged, config)
        return merged

    def _deep_update(self, base: Dict, update: Dict) -> Dict:
        """深度更新字典"""
        result = base.copy()
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_update(result[key], value)
            else:
                result[key] = value
        return result

    def load_with_inheritance(self, config_file: str) -> Dict[str, Any]:
        """加载配置文件（支持继承）"""
        config = self.load_yaml(config_file)

        # 处理继承
        if 'extends' in config:
            base_file = config['extends'] + '.yaml'
            base_config = self.load_with_inheritance(base_file)
            config = self.merge_configs(base_config, config)
            del config['extends']

        return config

    def load_experiment_config(self, experiment_name: str) -> Dict[str, Any]:
        """加载实验配置"""
        # 加载基础配置
        base_config = self.load_yaml("default.yaml")

        # 加载实验配置
        try:
            exp_config = self.load_yaml(f"experiments/{experiment_name}.yaml")
            final_config = self.merge_configs(base_config, exp_config)
        except FileNotFoundError:
            print(f"⚠️  实验配置不存在: {experiment_name}，使用基础配置")
            final_config = base_config

        return final_config

    def get_dataset_config(self, dataset_name: str) -> Dict[str, Any]:
        """获取数据集配置"""
        import tomlkit

        data_dir = self.config_dir.parent / "data"
        datasets_file = data_dir / "datasets.toml"

        if not datasets_file.exists():
            raise FileNotFoundError(f"数据集配置文件不存在: {datasets_file}")

        with open(datasets_file, 'r') as f:
            datasets = tomlkit.load(f)

        if dataset_name not in datasets:
            raise ValueError(f"数据集不存在: {dataset_name}")

        return datasets[dataset_name]

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """验证配置有效性"""
        required_keys = ['model', 'training']

        for key in required_keys:
            if key not in config:
                print(f"❌ 缺少必需的配置项: {key}")
                return False

        # 验证模型配置
        if 'd_model' not in config['model']:
            print("❌ 模型配置缺少 d_model")
            return False

        # 验证训练配置
        if 'batch_size' not in config['training']:
            print("❌ 训练配置缺少 batch_size")
            return False

        return True

    def save_config(self, config: Dict[str, Any], output_file: str):
        """保存配置到文件"""
        output_path = self.config_dir / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        print(f"✅ 配置已保存: {output_path}")


class TrainingConfig:
    """训练配置类 - 专门用于训练"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @classmethod
    def from_experiment(cls, experiment_name: str):
        """从实验名称创建配置"""
        loader = ConfigLoader()
        config = loader.load_experiment_config(experiment_name)
        return cls(config)

    @classmethod
    def from_args(cls, args):
        """从命令行参数创建配置"""
        config = {
            'model': {
                'd_model': args.d_model,
                'n_heads': args.n_heads,
                'n_layers': args.n_layers,
                'n_know': args.n_know,
                'dropout': args.dropout,
            },
            'llm': {
                'use_llm': getattr(args, 'use_llm', False),
                'llm_weight': getattr(args, 'llm_weight', 0.3),
                'pretrained_model': getattr(args, 'pretrained_model',
                                             'pretrained_models/qwen3-4b'),
            },
            'gnn': {
                'use_gnn': getattr(args, 'use_gnn', False),
                'n_kc': getattr(args, 'n_kc', 100),
                'gnn_layers': getattr(args, 'gnn_layers', 2),
            },
            'training': {
                'batch_size': args.batch_size,
                'test_batch_size': args.test_batch_size,
                'n_epochs': args.n_epochs,
                'learning_rate': args.learning_rate,
                'l2': getattr(args, 'l2', 1e-5),
                'early_stop': args.early_stop,
                'device': args.device,
            },
            'output': {
                'output_dir': args.output_dir,
                'from_file': getattr(args, 'from_file', None),
            }
        }
        return cls(config)

    def get(self, key_path: str, default=None):
        """获取配置值（支持路径）"""
        keys = key_path.split('.')
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def set(self, key_path: str, value: Any):
        """设置配置值（支持路径）"""
        keys = key_path.split('.')
        config = self.config

        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        config[keys[-1]] = value

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self.config.copy()

    def __getitem__(self, key: str) -> Any:
        """支持字典式访问"""
        return self.config.get(key)

    def __setitem__(self, key: str, value: Any):
        """支持字典式设置"""
        self.config[key] = value


# 便捷函数
def load_config(experiment_name: str) -> Dict[str, Any]:
    """加载实验配置"""
    loader = ConfigLoader()
    return loader.load_experiment_config(experiment_name)


def create_config_from_args(args) -> TrainingConfig:
    """从命令行参数创建配置"""
    return TrainingConfig.from_args(args)
