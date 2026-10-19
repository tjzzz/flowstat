#encoding=utf8
"""
Reader 基类说明（通用，去除原 aibee 专属）。

- 本地 reader: 继承 BaseReader，构造器 (exp_info)，实现 fetch_xxx(self) -> (pd.DataFrame, name)

reader.yaml 中注册: 文件名(去 .py): 类名
exp.yaml 中引用:  - reader: csv.load_events   （本地）
"""


class BaseReader(object):
    def __init__(self, exp_info=None):
        self.exp_info = exp_info
