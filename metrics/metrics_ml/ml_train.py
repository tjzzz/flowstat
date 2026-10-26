#encoding=utf8
"""传统 ML 训练 metric：从 DataFrame 训练一个分类/回归模型并产出评估。

移植自 aibee logstat 的 metrics/models/model_train.py，已脱敏。
配置（label_col / model_name / sparse_features / dense_features / train_test_method 等）
放在 par_dict；数据来自 FlowStat 的 df_dict（input[df_key]）或外部 CSV（par_dict['data_dir']）。
"""
import os
import json
import logging
import pickle

import pandas as pd
import numpy as np

from metrics.metrics_ml.ml_base import BaseClassifier
from metrics.metrics_ml.ml_utils import (
    get_input_data, get_job_id, perform_data_trans, get_train_test_data,
)


class MLTrainMetric(object):
    def process(self, input, par_dict=None):
        par_dict = par_dict or {}
        logging.basicConfig(level=logging.ERROR)

        label_col = par_dict['label_col']
        model_name = par_dict.get('model_name', 'lgb')
        sparse_features = par_dict.get('sparse_features', [])
        dense_features = par_dict.get('dense_features', [])
        feature_names = list(sparse_features) + list(dense_features)
        categorical_features_index = list(range(len(sparse_features)))

        df_features = get_input_data(input, par_dict, sep=par_dict.get('sep', '\t'))
        if df_features is None:
            raise ValueError('MLTrainMetric: 未找到建模数据，请在 par_dict 配置 df_key(指向 reader 产出) 或 data_dir。')

        MAX_TRAIN_NUM = int(par_dict.get('max_train_num', 4000000))
        if df_features.shape[0] > MAX_TRAIN_NUM:
            df_features = df_features.sample(MAX_TRAIN_NUM)

        uid = par_dict.get('uid')
        if uid and uid in df_features.columns:
            df_features = df_features.set_index(uid)

        job_id = par_dict.get('job_id', get_job_id())
        job_result_dir = par_dict.get('job_result_dir', 'job_result/' + job_id + '/')
        os.makedirs(job_result_dir, exist_ok=True)

        df_features, code_mapping_dict = perform_data_trans(
            df_features, label_col, categorical_features_index, feature_names, job_result_dir)

        if int(par_dict.get('return_df_features', 0)):
            return {'df_features': df_features}

        df_features = df_features[feature_names + [label_col]].dropna()
        train_test_input = {
            'train_test_method': par_dict.get('train_test_method', '0.8-0.2'),
            'train_start': par_dict.get('train_start'),
            'train_end': par_dict.get('train_end'),
            'balance_data': par_dict.get('balance_data', 0),
        }
        train, test = get_train_test_data(df_features, train_test_input, label_col)
        print(train[label_col].value_counts(), '\n', test[label_col].value_counts())

        trainX, trainY = train[feature_names], train[label_col]
        testX, testY = test[feature_names], test[label_col]

        instance = None
        for sc in BaseClassifier.__subclasses__():
            if sc.__name__ == model_name or getattr(sc({}), 'name', None) == model_name:
                instance = sc(par_dict)
                break
        if instance is None:
            raise ValueError('MLTrainMetric: 未找到模型 "%s"，可选: %s'
                             % (model_name, [getattr(c, 'name', c.__name__) for c in BaseClassifier.__subclasses__()]))

        instance.create_model(par_dict.get('model_par_dict', {}))
        if not par_dict.get('model_par_dict'):
            instance.tune_model(trainX, trainY, par_dict.get('tune_par_dict', {}))

        instance.model_fit(trainX, trainY)
        pred_test = instance.model_predict(testX)
        pred_train = instance.model_predict(trainX)

        perf_test = instance.evaluate_model(list(testY), pred_test, 'test')
        perf_train = instance.evaluate_model(list(trainY), pred_train, 'train')

        instance.save_result_model(instance.model, job_result_dir + 'save_' + model_name)

        model_result = {}
        model_result.update(perf_train)
        model_result.update(perf_test)
        model_result['pars'] = instance.get_model_pars()
        model_result.update({
            'train_num': int(train.shape[0]),
            'test_num': int(test.shape[0]),
            'feature_names': feature_names,
            'model_name': model_name,
        })
        with open(job_result_dir + 'model_result.json', 'w') as f:
            json.dump({k: v for k, v in model_result.items() if k != 'err_list'}, f)
        with open(job_result_dir + 'model_' + model_name + '.pkl', 'wb') as f:
            pickle.dump(instance.model, f)

        return {
            'model': instance.model,
            'eva_data': test[feature_names],
            'eva_label': test[label_col],
            'code_mapping_dict': code_mapping_dict,
            'categorical_features': categorical_features_index,
            'model_result': {'train': perf_train, 'test': perf_test},
            'feature_names': feature_names,
            'job_result_dir': job_result_dir,
        }
