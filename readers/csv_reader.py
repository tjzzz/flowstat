#encoding=utf8
"""示例本地 reader：从 examples/data 读取 CSV。可自行扩展其它数据源（DB/对象存储等）。"""
import os
import pandas as pd


class CSVReaders(object):
    def __init__(self, exp_info=None):
        self.exp_info = exp_info
        self.data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../examples/data')

    def load_events(self):
        """读取示例事件数据，返回 (DataFrame, 'events')。"""
        df = pd.read_csv(os.path.join(self.data_dir, 'sample_events.csv'))
        return df, 'events'

    def load_ab(self):
        """读取示例 A/B 实验数据(control/treatment)，返回 (DataFrame, 'ab')。"""
        df = pd.read_csv(os.path.join(self.data_dir, 'sample_ab.csv'))
        return df, 'ab'

    def load_ml(self):
        """读取示例分类建模数据(类别+连续特征+label)，返回 (DataFrame, 'ml')。"""
        df = pd.read_csv(os.path.join(self.data_dir, 'sample_ml.csv'))
        return df, 'ml'

    def load_ts(self):
        """读取示例时间序列数据(group_id/ds/y)，返回 (DataFrame, 'ts')。"""
        df = pd.read_csv(os.path.join(self.data_dir, 'sample_ts.csv'))
        return df, 'ts'
