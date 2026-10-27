#encoding=utf8
"""时间序列预测 metric（FlowStat 原生插件）。

移植自 aibee logstat 的 metrics/models/traffic_predict.py，已脱敏（去除店铺/MALL/内部目录约定）。
约定输入列：``group_id``（可选分组）、``ds``（日期）、``y``（目标值）；
数据来自 FlowStat 的 df_dict（input[df_key]）或外部 CSV（par_dict['data_dir']）。

输出：逐分组的回归评估（MAE/MAPE/RMSE/噪声检验）+ 整体加权汇总，并落盘 CSV 便于检视。
"""
import os
import logging

import pandas as pd

from metrics.metrics_ml.ml_evaluate import cal_model_performance_regress
from metrics.metrics_ml.ml_utils import get_input_data, get_job_id
from metrics.metrics_ts.ts_base import (
    GeneralTSPredict, LSTMPredict, VARPredict, test_white_noise, aggregate_results,
)


class TSPredictMetric(object):
    def process(self, input, par_dict=None):
        par_dict = par_dict or {}
        logging.basicConfig(level=logging.ERROR)

        model_par_dict = {
            'model_name': par_dict.get('model_name', input.get('model_name', 'base2')),
            'train_num': int(par_dict.get('train_num', 0)),
            'test_num': int(par_dict.get('test_num', 30)),
            'forcast_num': int(par_dict.get('forcast_num', par_dict.get('forcast_num', 1))),
            'eva_method': par_dict.get('eva_method', 'window'),
            'data_method': par_dict.get('data_method', 'seasonal'),
        }

        df = get_input_data(input, par_dict, sep=par_dict.get('sep', ','))
        if df is None:
            raise ValueError('TSPredictMetric: 未找到时序数据，请在 par_dict 配置 '
                             'df_key(指向 reader 产出) 或 data_dir。')
        if 'group_id' not in df.columns:
            df['group_id'] = 1

        modeler = GeneralTSPredict(model_par_dict)
        clean_data = modeler.preprocess(df)

        # 去除样本量过少的组（可选）
        min_samples = int(par_dict.get('min_samples', 0))
        if min_samples > 0:
            cnt = clean_data.groupby('group_id').count()
            bad = list(cnt[cnt['y'] < min_samples].index)
            clean_data = clean_data[~clean_data['group_id'].isin(bad)]

        group_id_list = list(set(clean_data['group_id']))
        print('Uniq group num =', len(group_id_list))

        model_name = model_par_dict['model_name']
        save_model = int(par_dict.get('save_model', 1))
        job_id = par_dict.get('job_id', get_job_id())
        result_dir = par_dict.get('result_dir', 'ts_result/' + job_id + '/')
        os.makedirs(result_dir, exist_ok=True)

        result_rows = []
        out_path = result_dir + 'result_%s.csv' % model_name
        with open(out_path, 'w') as ff:
            ff.write('group_id\tmethod\tMAE\tMAPE\tRMSE\tMAE_P\tNUM\tnoise\n')

            if model_name in ['VAR']:
                modeler2 = VARPredict(model_par_dict)
                model, real_list, pred_list = modeler2.model_VAR(clean_data)
                if save_model:
                    import pickle
                    pickle.dump(model, open(result_dir + 'model_%s.pkl' % model_name, 'wb'))
                out = cal_model_performance_regress(real_list, pred_list)
                ff.write('%s\t%s\t%s\n' % ('ALL', 'VAR', str(out)))
                result_rows.append(out)
            elif model_name == 'LSTM':
                modeler2 = LSTMPredict(model_par_dict)
                for m_group_id in group_id_list:
                    store_data = clean_data[clean_data['group_id'] == m_group_id]
                    store_data = store_data.sort_values(by='ds')
                    model, real_list, pred_list = modeler2.model_run(store_data)
                    out = cal_model_performance_regress(real_list, pred_list)
                    ff.write('%s\t%s\t%s\n' % (m_group_id, 'LSTM', str(out)))
                    result_rows.append(out)
            else:
                method_list = [model_name] if model_name != '-' else ['base2']
                for method in method_list:
                    modeler.model_name = method
                    for m_group_id in group_id_list:
                        store_data = clean_data[clean_data['group_id'] == m_group_id]
                        store_data = store_data.sort_values(by='ds')
                        model, real_list, pred_list = modeler.model_run(store_data)
                        if save_model and method in ['sarimax', 'prophet', 'LSTM']:
                            import pickle
                            pickle.dump(model, open(
                                result_dir + 'model_%s_%s.pkl' % (method, m_group_id), 'wb'))
                        out = cal_model_performance_regress(real_list, pred_list)
                        err_list = [pred_list[i] - real_list[i] for i in range(len(real_list))]
                        out['noise'] = test_white_noise(err_list)
                        ff.write('%s\t%s\t%.4f\t%.4f\t%.4f\t%.4f\t%d\t%s\n' % (
                            m_group_id, method, out['MAE'], out['MAPE'],
                            out['RMSE'], out['MAE_P'], out['NUM'], out['noise']))
                        result_rows.append(out)

        summary = aggregate_results(result_rows)
        with open(result_dir + 'summary.json', 'w') as f:
            import json
            json.dump(summary, f, indent=2)

        return {
            'result_rows': result_rows,
            'summary': summary,
            'result_dir': result_dir,
            'model_name': model_name,
            'group_num': len(group_id_list),
        }
