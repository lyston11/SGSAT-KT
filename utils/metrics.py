"""
Evaluation metrics for TriSG-KT
评估指标工具
"""

import numpy as np
from sklearn import metrics


def calculate_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics

    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities

    Returns:
        Dictionary of metrics
    """
    # Remove padding values
    mask = y_true >= 0
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    # Calculate metrics
    results = {
        'acc': metrics.accuracy_score(y_true, np.asarray(y_pred).round()),
        'auc': metrics.roc_auc_score(y_true, y_pred),
        'mae': metrics.mean_absolute_error(y_true, y_pred),
        'rmse': metrics.mean_squared_error(y_true, y_pred) ** 0.5,
    }

    return results


def print_metrics(metrics_dict: dict, prefix: str = ""):
    """Pretty print metrics"""
    print(f"\n{prefix}Evaluation Results:")
    print("-" * 50)
    for metric, value in metrics_dict.items():
        print(f"{metric.upper():10s}: {value:.4f}")
    print("-" * 50)
