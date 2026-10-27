"""滑动窗口基线预测器（通用）。

移植自 aibee logstat 的 metrics/models/sale_predict/window.py，已脱敏。
最简单的时序基线：用历史窗口均值作为下一个点的预测。
"""
import numpy as np
import pandas as pd


class Window(object):
    def __init__(self, df, label, window_size=7):
        """
        :param df: 需要处理的 DataFrame
        :param label: 预测值所在的列名
        :param window_size: 滑动窗口大小
        """
        self.df = df
        self.label = label
        self.window_size = window_size

    def predict(self, window_size=None):
        """返回每个位置用其之前 window_size 个值均值得到的预测列表。"""
        if window_size is None:
            window_size = self.window_size
        value_list = self.df[self.label].tolist()
        value_predict_list = []
        for i in range(len(value_list)):
            start = max(0, i - window_size)
            temp = value_list[start:i]
            value_predict = np.mean(temp) if temp else value_list[i]
            value_predict_list.append(value_predict)
        return value_predict_list


if __name__ == '__main__':
    data = pd.DataFrame(range(1000))
    data.columns = ['data']
    win = Window(data, 'data')
    preds = win.predict()
