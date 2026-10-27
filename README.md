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
- `metrics/`         metric 插件库（自带 `metrics_demo` / `metrics_ab` / `metrics_ml`）
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

## A/B 实验支持

`src/abtools.py` 提供通用统计函数，`metrics/metrics_ab/` 提供框架原生插件：

- `srm_test`              样本比例失衡(SRM)卡方检验
- `welch_ttest`           Welch 两样本 t 检验（均值差 + 置信区间 + Cohen's d）
- `two_proportion_test`   两比例 z 检验（转化率）
- `min_sample_size_mean`        连续指标最小样本量（输入 CV + 相对提升）
- `min_sample_size_proportion` 比例指标最小样本量（基线 + 绝对 MDE）
- `cuped_adjust`          CUPED 方差缩减（用实验前协变量降低方差）
- `benjamini_hochberg`   多指标同时检验时的 FDR 校正

跑示例：

```bash
python run.py -e examples/3_ab_analysis.yaml -m ab_analysis
python run.py -e examples/4_sample_size.yaml -m plan_sample
```

## 传统机器学习建模（metrics_ml）

`metrics/metrics_ml/` 是从通用建模场景中抽象出的监督学习工具（移植自内部项目，已脱敏），提供统一接口的模型基类与一键训练/预测/解释/特征分析/对比：

- `ml_base.BaseClassifier`        统一基类 + LR / LGBM / XGB / RF / KNN / DT / SGD / AdaBoost 封装（FM/DeepFM 需 deepctr，可选）
- `MLTrainMetric`           训练一个分类/回归模型，自动编码类别特征、切分、调参、评估、落盘 pkl
- `MLPredictMetric`         加载训练产出的编码映射，对新数据预测并评估
- `MLExplainMetric`         SHAP 特征归因 + 按维度拆解定位预测错误（卡方 / t 检验）
- `FeatureAnalysisMetric`   自动特征显著性分析（卡方 / 方差分析 / 皮尔逊 / 序列）
- `ModelReportMetric`       汇总多个训练 job 的评估指标做对比

跑示例（默认 `rf`，仅依赖 scikit-learn）：

```bash
python run.py -e examples/5_ml_train.yaml -m ml_train
```

`MLTrainMetric` 的 par_dict 关键项：`df_key`(reader 产出的 DataFrame 名)、`label_col`、`model_name`、`sparse_features`(类别)、`dense_features`(连续)、`train_test_method`(比例或按时间)、`job_result_dir`。

## 时间序列预测（metrics_ts）

`metrics/metrics_ts/` 是从通用时序预测场景中抽象出的工具（移植自内部项目，已脱敏），支持单序列/多分组序列的滑动预测与多种模型：

- `ts_base.GeneralTSPredict`  通用入口，内置 base / base2（朴素） / prophet / sarimax / sarimax_garch / sarimax_ex / lgb 等模型，支持 raw|seasonal|seasonal_diff 量纲变换
- `ts_base.LSTMPredict`       基于 keras 的单序列 LSTM 预测（懒加载）
- `ts_base.VARPredict`        基于 statsmodels VAR 的多序列联合预测（懒加载）
- `window.Window`             滑动窗口均值基线
- `prophet_model.ProphetForecaster`  带外生回归量 + 中国节假日的 Prophet 封装
- `xgb_ar.XgbARForecaster`    用历史滞后值做特征的 XGBoost 自回归
- `TSPredictMetric`           框架原生插件，逐分组跑预测并输出 MAE/MAPE/RMSE + 整体加权汇总

跑示例（默认 `base2`，仅需 pandas/numpy；prophet/sarimax/lgb 等按需装对应依赖）：

```bash
python run.py -e examples/6_ts_predict.yaml -m ts_predict
```

`TSPredictMetric` 的 par_dict 关键项：`df_key`(reader 产出的 DataFrame 名，含 group_id/ds/y)、`model_name`、`data_method`、`train_num`、`test_num`、`forcast_num`、`min_samples`(过滤样本过少组)、`result_dir`。

## 提交前自检

- [x] 无明文密码 / keytab
- [x] 无内部域名 / 内网 IP
- [x] 无真实客户/商场数据（示例用合成数据）
- [x] `.gitignore` 已忽略 `.DS_Store` / `__pycache__` / `*.keytab`
