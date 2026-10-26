"""模型评估指标：二分类与回归。纯函数，框架无关，可单独 import。

移植自 aibee logstat 的 metrics/models/evaluate.py，已脱敏（去除内部域名/凭证/业务）。
"""
import numpy as np
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, precision_score,
    recall_score, accuracy_score,
)


def cal_model_performance_binary(y_real, y_pred, prefix=''):
    """二分类评估。y_real/y_pred 为 0/1 列表，prefix 用于区分 train/test。"""
    zz = confusion_matrix(y_real, y_pred, labels=[1, 0])
    if prefix != '':
        prefix = prefix + '_'

    model_result = {
        prefix + 'AUC': roc_auc_score(y_real, y_pred),
        prefix + 'confusion_matrix': zz.tolist(),
        prefix + 'precision_score': round(precision_score(y_real, y_pred), 3),
        prefix + 'recall_score': round(recall_score(y_real, y_pred), 3),
        prefix + 'accuracy_score': round(accuracy_score(y_real, y_pred), 3),
    }
    return model_result


def cal_model_performance_regress(y_real, y_pred, prefix=''):
    """回归评估：MAE / MAPE / RMSE / 均值相对误差。

    注：原实现对 y_real==0 做 +1 平移以避免除零，这里保留该口径以保证与历史口径一致。
    """
    y_real = list(y_real)
    y_pred = list(y_pred)
    model_result = {}
    DELTA = 1
    for i in range(len(y_real)):
        if y_real[i] == 0:
            y_real[i] = DELTA
            y_pred[i] += DELTA

    err_list = [y_pred[i] - y_real[i] for i in range(len(y_real))]
    err_rate_list = [abs(err_list[i] / y_real[i]) for i in range(len(y_real))]
    mae = abs(np.array(err_list)).mean()
    mape = np.array(err_rate_list).mean()
    mse = sum([err ** 2 for err in err_list]) / len(err_list)
    rmse = mse ** 0.5

    if prefix != '':
        prefix = prefix + '_'
    model_result.update({
        prefix + 'MAE': mae,
        prefix + 'MAPE': mape,
        prefix + 'RMSE': rmse,
        prefix + 'NUM': len(y_real),
        prefix + 'MAE_P': mae / np.array(y_real).mean(),
        prefix + 'mean_y': int(sum(y_real) / len(y_real)),
        prefix + 'err_list': err_list,
    })
    return model_result
