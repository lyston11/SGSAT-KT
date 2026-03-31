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
