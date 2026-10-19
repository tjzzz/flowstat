# FlowStat

基于 DataFrame 的**通用日志/事件统计分析框架**（插件化 metric + reader 注册 + yaml 驱动实验）。

## 架构

```
exp.yaml ──▶ ExpConf 解析
                │
                ▼
          StreamTask 按类名扫描 metric 目录，链式实例化
                │
   reader(本地 CSV) 提供 DataFrame ──▶ metric.process(df_dict, pars) 深度合并 ──▶ 输出
```

- `run.py`            总入口（CLI + `run()` 函数）
- `utils.py`          `ExpConf`（解析 yaml）、`StreamTask`（加载并链式执行 metric）
- `readers/`         reader 注册（`reader.yaml`）+ `BaseReader` 基类 + 示例 `CSVReaders`
- `metrics/`         metric 插件库（自带 `metrics_demo` 示例）
- `src/`             通用工具（`tools` 日期/几何、`send_mail` 邮件占位）
- `examples/`        示例 yaml + 合成数据

## 安装

```bash
pip install -r requirements.txt
# 停留点检测: pyinter；轨迹长度: shapely
```

## 快速开始

```bash
cd flowstat
python run.py -e examples/1_read_data.yaml -m demo1_read_data
```

输出：`事件数= 4`（示例数据中 floor=2F 共 4 条）。

或在代码中调用：

```python
from run import run
df_out = run('examples/1_read_data.yaml', 'demo1_read_data', {})
```

## 写一个新 metric

在 `metrics/your_pkg/your_metric.py` 中定义类，实现 `process`：

```python
class MyMetric(object):
    def process(self, input, par_dict=None):
        events = input['events']
        # ... 你的逻辑 ...
        return {'events': events, 'my_result': ...}
```

在 `exp.yaml` 的 stream 下引用（类名即 key）：

```yaml
conf:
    metric_dir: 'metrics/your_pkg/'
my_stream:
    - reader: csv.load_events
    - MyMetric: 'some_par:val'
```

## 写一个新 reader

继承 `BaseReader`，实现 `fetch_xxx(self)` 返回 `(DataFrame, name)`，
并在 `readers/reader.yaml` 注册：`your_reader: YourReaders`。

## 提交前自检

- [x] 无明文密码 / keytab
- [x] 无内部域名 / 内网 IP
- [x] 无真实客户/商场数据（示例用合成数据）
- [x] `.gitignore` 已忽略 `.DS_Store` / `__pycache__` / `*.keytab`
