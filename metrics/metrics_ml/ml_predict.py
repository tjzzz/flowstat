#encoding=utf8
"""传统 ML 预测 metric：加载训练产出的 code_mapping_dict，对新数据编码后预测并评估。

移植自 aibee logstat 的 metrics/models/model_predict.py，已脱敏。
依赖 ml_train.perform_data_trans（保证与训练一致的编码）。
"""
import logging

from metrics.metrics_ml.ml_evaluate import cal_model_performance_binary
from metrics.metrics_ml.ml_utils import get_input_data, perform_data_trans


class MLPredictMetric(object):
    def process(self, input, par_dict=None):
        par_dict = par_dict or {}
        logging.basicConfig(level=logging.ERROR)

        label_col = par_dict['label_col']
        sparse_features = par_dict.get('sparse_features', [])
        dense_features = par_dict.get('dense_features', [])
        feature_names = list(sparse_features) + list(dense_features)
        categorical_features_index = list(range(len(sparse_features)))

        df_features = get_input_data(input, par_dict, sep=par_dict.get('sep', '\t'))
        if df_features is None:
            raise ValueError('MLPredictMetric: 未找到预测数据。')

        uid = par_dict.get('uid')
        if uid and uid in df_features.columns:
            df_features = df_features.set_index(uid)

        job_result_dir = par_dict['job_result_dir']
        df_features, code_mapping_dict = perform_data_trans(
            df_features, label_col, categorical_features_index, feature_names, job_result_dir)

        model = input['model']
        pred_score = model.predict(df_features[feature_names])
        pred_res = [int(score > 0.5) for score in pred_score]
        real_res = list(df_features[label_col])
        res = cal_model_performance_binary(real_res, pred_res)
        print(res)

        return {'df_features': df_features, 'pred_result': res}
