"""时间序列预测核心引擎（通用）。

移植自 aibee logstat 的 metrics/models/ts_predict.py，已脱敏（去除内部业务/店铺/MALL 概念）。
提供三类预测器与若干纯函数工具：

- `GeneralTSPredict`  ：单序列/多分组序列的通用滑动预测入口，内置 base / base2 / prophet /
                       sarimax / sarimax_garch / sarimax_ex / lgb 等模型
- `LSTMPredict`       ：基于 keras 的单序列 LSTM 预测（懒加载）
- `VARPredict`        ：基于 statsmodels VAR 的多序列联合预测（懒加载）

重依赖（prophet / statsmodels / arch / lightgbm / keras）均按需懒加载，
未安装时仅在调用对应模型时报错，不影响其它模型与框架导入。
"""
import datetime
import itertools
import math

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# 纯函数工具
# --------------------------------------------------------------------------- #
def get_day_of_week(dt):
    """返回日期对应的星期（0=周一）。dt 可为 Timestamp / datetime / date。"""
    try:
        return dt.weekday()
    except AttributeError:
        return pd.to_datetime(dt).weekday()


def decode_data_trans(data_method, raw_data, new_list):
    """把预处理后的预测值还原回原始量纲。

    - raw      : 不做变换，原样返回
    - seasonal : 乘回周期因子 rate
    - seasonal_diff : 周期因子 + 一阶差分还原
    """
    n_test = len(new_list)
    if data_method == 'raw':
        return new_list
    if data_method == 'seasonal':
        rate_list = list(raw_data[-n_test:]['rate'])
        return [new_list[i] * rate_list[i] for i in range(len(new_list))]
    if data_method == 'seasonal_diff':  # season + diff
        raw_list = []
        tmp_data = raw_data[-(n_test + 1):]
        for i in range(n_test):
            res = (tmp_data.iloc[i]['y'] + new_list[i]) * tmp_data.iloc[i + 1]['rate']
            raw_list.append(res)
        return raw_list
    return new_list


def test_white_noise(err_list):
    """对模型残差做 Ljung-Box 白噪声检验，判断是否已提取足够信息。"""
    from statsmodels.stats.diagnostic import acorr_ljungbox  # 懒加载

    if len(err_list) > 2:
        alpha = 0.05
        res = acorr_ljungbox(err_list, lags=min(7, len(err_list) - 2))
        # 兼容返回 DataFrame 或 (stat, pvalue) 元组两种形态
        if hasattr(res, 'columns'):  # DataFrame（新版 statsmodels 默认）
            p_value_list = res.iloc[:, 1].tolist() if res.shape[1] > 1 else res.iloc[:, 0].tolist()
        elif isinstance(res, tuple):
            p_value_list = res[1]
        else:
            p_value_list = list(res)
        # 跳过 NaN，统计显著（p < alpha）的滞后数
        sig_num = sum((isinstance(p, (int, float)) and p == p and p < alpha) for p in p_value_list)
        return sig_num == 0
    return True


def grid_search(ts):
    """SARIMA 参数网格搜索（按 BIC 选优），固定 7 步季节性周期。"""
    from statsmodels.tsa.statespace.sarimax import SARIMAX  # 懒加载

    p = d = q = range(0, 3)
    pdq = list(itertools.product(p, d, q))
    pdq_x_PDQs = [(0, 0, 0, 7)]
    a, b, c = [], [], []
    for param in pdq:
        for seasonal_param in pdq_x_PDQs:
            try:
                mod = SARIMAX(ts, order=param, seasonal_order=seasonal_param,
                              enforce_stationarity=False, enforce_invertibility=False)
                results = mod.fit(disp=False)
                a.append(param)
                b.append(seasonal_param)
                c.append(results.bic)
            except Exception:
                continue
    wf = pd.DataFrame({'pdq': a, 'pdq_x_PDQs': b, 'bic': c})
    if wf.empty:
        return (0, 0, 0), (0, 0, 0, 7)
    tmp = wf[wf['bic'] == wf['bic'].min()].iloc[0]
    return tmp['pdq'], tmp['pdq_x_PDQs']


def get_ex_features(y):
    """构造滚动统计外生特征（最近 7 步的 min/max/std/mean/median）。"""
    tmp = np.array(y)
    all_res = []
    for i in range(len(tmp)):
        ts = tmp[:i + 1]
        if len(ts) > 7:
            ts = ts[-7:]
        all_res.append([ts.min(), ts.max(), ts.std(), ts.mean(), np.median(ts)])
    return all_res[:-1], [all_res[-1]]


def create_dataset(dataset, look_back=1):
    """构造 LSTM 监督学习样本（单步预测）。"""
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        dataX.append(dataset[i:(i + look_back), 0])
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


def aggregate_results(result_rows):
    """对多分组的结果做整体加权平均（MAE/MAPE/RMSE/MAE_P）。

    result_rows: list[dict]，每项含 MAE/MAPE/RMSE/NUM/MAE_P/mean_y
    返回整体 dict（与单组口径一致：用样本量加权）。
    """
    if not result_rows:
        return {}
    total_num = sum(r.get('NUM', 0) for r in result_rows)
    if total_num == 0:
        return {}
    out = {'MAE': 0.0, 'MAPE': 0.0, 'RMSE': 0.0, 'MAE_P': 0.0, 'mean_y': 0.0, 'NUM': total_num}
    for r in result_rows:
        n = r.get('NUM', 0)
        out['MAE'] += r.get('MAE', 0) * n
        out['MAPE'] += r.get('MAPE', 0) * n
        out['RMSE'] += r.get('RMSE', 0) ** 2 * n
        out['mean_y'] += r.get('mean_y', 0) * n
    out['MAE'] /= total_num
    out['MAPE'] /= total_num
    out['RMSE'] = math.sqrt(out['RMSE'] / total_num)
    out['mean_y'] /= total_num
    out['MAE_P'] = out['MAE'] / (out['mean_y'] or 1)
    return out


# --------------------------------------------------------------------------- #
# GeneralTSPredict
# --------------------------------------------------------------------------- #
class GeneralTSPredict(object):
    """时间序列相关的预测函数（单序列/多分组）。

    输入列约定：``ds``（日期）、``y``（目标值）、``group_id``（可选分组）。
    par_dict 关键项：model_name / train_num / test_num / forcast_num / eva_method /
    data_method（raw|seasonal|seasonal_diff）。
    """

    def __init__(self, par_dict):
        self.model_name = par_dict.get('model_name', '-')
        self.train_num = int(par_dict.get('train_num', 0))
        self.test_num = int(par_dict.get('test_num', 30))
        self.eva_method = par_dict.get('eva_method', 'window')
        self.data_method = par_dict.get('data_method', 'raw')
        self.forcast_num = int(par_dict.get('forcast_num', 1))

    def preprocess(self, data):
        """缺失插补在外部完成；此处计算周期因子 rate 并做可选量纲变换。"""
        data = data.copy()
        data['ds'] = pd.to_datetime(data['ds'], errors='coerce')
        data['day_of_week'] = data['ds'].apply(get_day_of_week)

        group_data = data.groupby(['group_id', 'day_of_week']).mean()['y'].reset_index()
        base_data = group_data[group_data['day_of_week'] == 0][['group_id', 'y']]
        group_data = pd.merge(group_data, base_data, on=['group_id'], how='left')
        group_data['rate'] = group_data['y_x'] / group_data['y_y']
        data = pd.merge(data, group_data, on=['group_id', 'day_of_week'], how='left')

        data['y_bak'] = data['y']

        if self.data_method == 'seasonal':
            data['y'] = data['y'] / data['rate']
        elif self.data_method == 'seasonal_diff':
            tmp = data['y'] / data['rate']
            data['y'] = tmp.diff(1)

        data = data.drop(columns=['day_of_week', 'y_x', 'y_y'], errors='ignore')
        return data

    def model_run(self, data):
        """统一的滑动预测入口。data 已按 ds 排好序。"""
        n_train = self.train_num
        n_test = self.test_num
        model_name = self.model_name
        forcast_num = self.forcast_num

        pred_list_raw, real_list_raw = [], []
        n_eva = n_test + forcast_num - 1

        for _ in range(n_test):
            train = data[:-n_eva].dropna()
            if n_train != 0:
                train = train.tail(n_train)
            test = data[-n_eva:].head(forcast_num)

            model_func = getattr(self, 'model_' + model_name, None)
            if model_func is None:
                raise ValueError('GeneralTSPredict: 未知模型 "%s"' % model_name)
            model, tmp_pred_list = model_func(train, test)

            n_eva -= 1
            pred_list_raw += decode_data_trans(self.data_method, test, tmp_pred_list)
            real_list_raw += decode_data_trans(self.data_method, test, test['y'].tolist())

        return model, real_list_raw, pred_list_raw

    # ---- 基准模型 ----
    def model_base(self, df, pred_df):
        forcast_num = pred_df.shape[0]
        window = 7
        last_window_list = list(df[-window:]['y'])
        pred_list = []
        for _ in range(forcast_num):
            mean_value = sum(last_window_list[-window:]) / 7
            pred_list.append(mean_value)
            last_window_list.append(mean_value)
        return 'base', pred_list

    def model_base2(self, df, pred_df):
        forcast_num = pred_df.shape[0]
        return 'base2', forcast_num * [float(np.array(df['y'])[-1])]

    # ---- Prophet ----
    def model_prophet(self, train_data, pred_df):
        try:
            from prophet import Prophet
        except ImportError:
            from fbprophet import Prophet  # 兼容旧包名

        forcast_num = pred_df.shape[0]
        m = Prophet(seasonality_mode='multiplicative', seasonality_prior_scale=1,
                    yearly_seasonality=False, weekly_seasonality=True)
        m.add_country_holidays(country_name='CN')
        for column in train_data.columns:
            if column not in ['group_id', 'ds', 'y', 'rate', 'y_bak']:
                m.add_regressor(column, mode='multiplicative')
        m.fit(train_data)
        forecast = m.predict(pred_df)
        yhat_list = list(forecast[-forcast_num:]['yhat'])
        return m, yhat_list

    # ---- GARCH 残差修正 ----
    def model_garch(self, err_list, forcast_num=1):
        from arch import arch_model  # 懒加载
        last_obs = 30
        am = arch_model(err_list)
        res = am.fit(last_obs=last_obs, disp='off')
        tmp = res.forecast(horizon=3)
        return tmp.mean.iloc[-1]['h.1']

    # ---- SARIMAX ----
    def model_sarimax(self, df, pred_df):
        from statsmodels.tsa.statespace.sarimax import SARIMAX  # 懒加载
        forcast_num = pred_df.shape[0]
        df = df.reset_index(drop=True)
        pdq, pdq_x = grid_search(df['y'])
        print('best (p,d,q)=%s seasonal=%s' % (str(pdq), str(pdq_x)))
        model = SARIMAX(df['y'], order=pdq, seasonal_order=pdq_x,
                        enforce_stationarity=False, enforce_invertibility=False)
        model_fit = model.fit(disp=False)
        yhat = model_fit.predict(start=len(df), end=len(df) + forcast_num - 1)
        return model_fit, list(yhat)

    def model_sarimax_garch(self, df, pred_df):
        forcast_num = pred_df.shape[0]
        model_fit, pred_res = self.model_sarimax(df, pred_df)
        pred_list = list(model_fit.predict(1, len(df)))
        real_list = list(df['y'])
        err_list = [pred_list[i] - real_list[i] for i in range(len(real_list))]
        # 检验残差平方是否存在自相关（异方差）
        from statsmodels.tsa.stattools import acf  # 懒加载
        m = 7
        _, _, _, p = acf(err_list, nlags=m, fft=False, qstat=True)
        if p[0] < 0.05:
            garch_mean = self.model_garch(err_list)
            pred_res = [r + garch_mean for r in pred_res]
        return model_fit, pred_res

    def model_sarimax_ex(self, df, forcast_num):
        from statsmodels.tsa.statespace.sarimax import SARIMAX  # 懒加载
        pdq, pdq_x = grid_search(df['y'])
        ex_features, ex_forecast = get_ex_features(df['y'])
        model = SARIMAX(endog=df['y'][1:], exog=ex_features, order=pdq,
                        seasonal_order=pdq_x, enforce_stationarity=False,
                        enforce_invertibility=False)
        model_fit = model.fit(disp=False)
        yhat = model_fit.predict(len(df) - 1, len(df) - 1, exog=ex_forecast)
        return yhat.iloc[0]

    # ---- LightGBM 监督回归（lag 特征） ----
    def model_lgb(self, df, pred_df):
        import lightgbm  # 懒加载
        model = lightgbm.LGBMRegressor(objective='regression', num_leaves=10,
                                       learning_rate=0.01, n_estimators=100, max_depth=4)
        model.fit(df)
        pred_res = model.predict(pred_df)
        return model, list(pred_res)


# --------------------------------------------------------------------------- #
# LSTMPredict
# --------------------------------------------------------------------------- #
class LSTMPredict(object):
    """基于 keras LSTM 的单序列预测（懒加载 keras）。"""

    def __init__(self, par_dict):
        self.train_num = int(par_dict.get('train_num', 0))
        self.test_num = int(par_dict.get('test_num', 30))
        self.eva_method = par_dict.get('eva_method', 'window')

    def preprocess(self, store_data):
        from sklearn.preprocessing import MinMaxScaler  # 懒加载

        store_data = store_data.sort_values(by='ds')
        dataset = store_data[['y']].values.astype('float32')
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)

        train_size = self.train_num
        test_size = self.test_num
        if train_size == 0:
            train_size = len(dataset) - test_size
        dataset = dataset[-(train_size + test_size):]
        train, test = dataset[:-test_size], dataset[-test_size:]

        look_back = 1
        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)
        trainX = np.reshape(trainX, (trainX.shape[0], look_back, trainX.shape[1]))
        testX = np.reshape(testX, (testX.shape[0], look_back, testX.shape[1]))
        return dataset, trainX, trainY, testX, testY, scaler

    def model_lstm(self, trainX, trainY, testX, testY):
        from keras.models import Sequential
        from keras.layers import Dense, LSTM
        from keras.callbacks import EarlyStopping  # 懒加载

        earlystop = EarlyStopping(monitor='loss', patience=50)
        model = Sequential()
        model.add(LSTM(50, input_shape=(trainX.shape[1], trainX.shape[2])))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(trainX, trainY, epochs=100, callbacks=[earlystop],
                  batch_size=1, verbose=0)
        train_predict = model.predict(trainX)
        test_predict = model.predict(testX)
        return model, train_predict, test_predict

    def model_run(self, store_data):
        dataset, trainX, trainY, testX, testY, scaler = self.preprocess(store_data)
        model, _, test_predict = self.model_lstm(trainX, trainY, testX, testY)
        test_predict = scaler.inverse_transform(test_predict)
        testY = scaler.inverse_transform([testY])[0]
        return model, list(testY), list(test_predict[:, 0])


# --------------------------------------------------------------------------- #
# VARPredict
# --------------------------------------------------------------------------- #
class VARPredict(object):
    """基于 statsmodels VAR 的多序列联合预测（懒加载 statsmodels）。"""

    def __init__(self, par_dict):
        self.model_name = par_dict.get('model_name', '-')
        self.train_num = int(par_dict.get('train_num', 0))
        self.test_num = int(par_dict.get('test_num', 30))
        self.eva_method = par_dict.get('eva_method', 'window')
        self.data_method = par_dict.get('data_method', 'raw')

    def preprocess(self, all_data):
        data = pd.pivot_table(all_data, index='ds', columns='group_id',
                              values='y', aggfunc=[np.sum])['sum']
        data = data.fillna(1)
        data_rate = pd.pivot_table(all_data, index='ds', columns='group_id',
                                   values='rate', aggfunc=[np.sum])['sum']
        data_rate = data_rate.fillna(1)
        return data, data_rate

    def model_VAR(self, all_data):
        from statsmodels.tsa.api import VAR  # 懒加载

        data, data_rate = self.preprocess(all_data)
        n_train = self.train_num
        n_test = self.test_num
        lag_num = 14
        test_rate = data_rate[-n_test:]
        a, b = test_rate.shape
        rate_list = np.array(test_rate.values).reshape(1, a * b)[0]

        pred_list, real_list = [], []
        train = data[:-n_test]
        test = data[-n_test:]
        history = train.tail(n_train) if n_train else train
        for i in range(len(test)):
            model = VAR(history)
            results = model.fit(lag_num, trend="nc")
            res = results.forecast(history.values, 1)
            pred_list += list(res[0])
            real_list += list(test.iloc[i])
            history = pd.concat([train, test.head(i + 1)]).tail(n_train or len(train))

        if len(rate_list) > 0:
            real_list = [real_list[i] * rate_list[i] for i in range(len(real_list))]
            pred_list = [pred_list[i] * rate_list[i] for i in range(len(pred_list))]
        return model, real_list, pred_list
