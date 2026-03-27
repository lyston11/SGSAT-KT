"""
SGSAT-KT 训练配置文件
在这里修改参数，不用动代码
"""

EXPERIMENTS = {
    'test': {
        'description': '快速测试 - 验证代码修改',
        'dataset': 'algebra05',
        'epochs': 5,
        'batch_size': 32,
        'learning_rate': 0.001,
        'device': 'cuda',
        'use_llm': False,
        'use_gnn': False,
        'use_precomputed': False,
    },

    'baseline': {
        'description': '基线模型 - 无任何增强',
        'dataset': 'xes',
        'epochs': 100,
        'batch_size': 64,
        'learning_rate': 0.001,
        'device': 'cuda',
        'use_llm': False,
        'use_gnn': False,
        'use_precomputed': False,
    },

    'full': {
        'description': '完整模型 - LLM + GNN + 预计算嵌入',
        'dataset': 'xes',
        'epochs': 100,
        'batch_size': 64,
        'learning_rate': 0.001,
        'device': 'cuda',
        'use_llm': True,
        'use_gnn': True,
        'use_precomputed': True,
        'n_kc': 812,
        'gnn_layers': 2,
    },

    'prod': {
        'description': '生产环境 - 最佳性能配置',
        'dataset': 'xes',
        'epochs': 200,
        'batch_size': 128,
        'learning_rate': 0.0005,
        'device': 'cuda',
        'use_llm': True,
        'use_gnn': True,
        'use_precomputed': True,
        'n_kc': 812,
        'gnn_layers': 3,
    },
}

DEFAULT_CONFIG = {
    'd_model': 128,
    'n_heads': 8,
    'n_layers': 2,
    'n_know': 32,
    'dropout': 0.2,
    'l2': 1e-5,
    'early_stop': 10,
    'test_batch_size': 8,
    'save_model': True,
    'pretrained_model': 'pretrained_models/Qwen3-Embedding-4B',
}
