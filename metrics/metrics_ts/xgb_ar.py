"""XGBoost 自回归预测封装（通用）。

移植自 aibee logstat 的 metrics/models/sale_predict/xgboost_predict.py，已脱敏。
思路：用历史 N 步的滞后值（value_1..value_N）与可选类别特征构造监督样本，
训练一个回归 XGBoost 预测下一步目标值。
"""
import pandas as pd


class XgbARForecaster(object):
    def __init__(self):
        self.model = None

    def train(self, train_data, target_label, cate_col=None):
        """训练自回归 XGBoost。

        :param train_data: 含目标列与可选类别列的 DataFrame（按时间有序）
        :param target_label: 需要预测的列名
        :param cate_col: 需要做 one-hot 的类别列
        """
        import xgboost as xgb  # 懒加载：可选依赖
        cate_col = cate_col or []
        squared_error = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'tree_method': 'hist',
        }
        feature, label = self.generate_feature(train_data, target_label, cate_col)
        dtrain = xgb.DMatrix(feature, label)
        bst = xgb.train(squared_error, dtrain, 30)
        self.model = bst
        return bst

    def predict(self, data):
        if self.model is None:
            raise RuntimeError('XgbARForecaster: 请先调用 train()。')
        dtotal = xgb.DMatrix(data)
        return self.model.predict(dtotal)

    def generate_feature(self, data, label, cate_col, time_shift=7):
        """构造滞后特征。

        :param time_shift: 用多少历史步做特征
        """
        df = data[label].to_frame()
        col = []
        for i in range(1, time_shift + 1):
            df['value_%d' % i] = data[label].shift(i)
            col.append('value_%d' % i)
        col.extend(cate_col)
        data_process = df[col + [label]].copy()
        dummie = pd.get_dummies(data_process, columns=cate_col)
        dummie = dummie.dropna()
        target = dummie[label]
        dummie = dummie.drop([label], axis=1)
        return dummie, target


if __name__ == '__main__':
    import numpy as np
    n = 200
    data = pd.DataFrame({'y': np.sin(np.arange(n) / 5.0) + np.random.normal(0, 0.1, n)})
    forecaster = XgbARForecaster()
    forecaster.train(data, 'y', [])
    feat, lbl = forecaster.generate_feature(data, 'y', [])
    preds = forecaster.predict(feat)
    print('pred sample:', preds[:5])
