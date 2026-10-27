"""Prophet 预测封装（通用，懒加载 prophet）。

移植自 aibee logstat 的 metrics/models/sale_predict/prophet.py，已脱敏（去除商品/销量/天气等业务列）。
对外暴露通用的「带回归量的 Prophet 训练 / 预测」接口：
- 任意 DataFrame 需包含 ``ds``（日期）、``y``（目标值）以及可选的外生回归量列
- 可选开启中国法定节假日效应
"""
import pandas as pd


class ProphetForecaster(object):
    def __init__(self, para_dict=None):
        self.model = None
        self.para_dict = para_dict or {}

    def train(self, df, cn_holidays=True, holidays_df=None,
              regressor_mode='multiplicative'):
        """训练 Prophet 模型。

        :param df: 含 ds / y 及可选外生列的 DataFrame（外生列会自动 add_regressor）
        :param cn_holidays: 是否加入中国法定节假日效应
        :param holidays_df: 自定义节假日表（需含 ds / holiday 列）
        :param regressor_mode: 回归量作用方式（'multiplicative' | 'additive'）
        """
        try:
            from prophet import Prophet
        except ImportError:
            from fbprophet import Prophet  # 兼容旧包名

        if cn_holidays:
            model = Prophet(daily_seasonality=False)
            model.add_country_holidays(country_name='CN')
        elif holidays_df is not None and not holidays_df.empty:
            model = Prophet(daily_seasonality=False, holidays=holidays_df)
        else:
            model = Prophet(daily_seasonality=False)

        for column in df.columns:
            if column not in ['ds', 'y']:
                model.add_regressor(column, mode=regressor_mode)

        model.fit(df)
        self.model = model
        return model

    def predict(self, predict_df, need_round=False, clip_lower=0):
        """对 predict_df（含 ds 与相同外生列）做预测，可选取整/下限裁剪。"""
        if self.model is None:
            raise RuntimeError('ProphetForecaster: 请先调用 train()。')
        result_df = self.model.predict(predict_df)
        if need_round:
            result_df['predict_num'] = result_df['yhat'].round().clip(
                lower=clip_lower).astype(int)
        return result_df


if __name__ == '__main__':
    import numpy as np
    dates = pd.date_range('2023-01-01', periods=365, freq='D')
    df = pd.DataFrame({'ds': dates, 'y': 100 + 10 * np.sin(np.arange(365) / 7.0)})
    forecaster = ProphetForecaster()
    forecaster.train(df, cn_holidays=True)
    future = forecaster.model.make_future_dataframe(periods=30)
    result = forecaster.predict(future)
    print(result[['ds', 'yhat']].tail())
