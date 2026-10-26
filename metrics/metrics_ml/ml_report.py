#encoding=utf8
"""模型对比 metric：汇总多个训练 job 的评估指标到一个 DataFrame。

移植自 aibee logstat 的 metrics/models/model_report.py，已脱敏（job 目录可配置）。
"""
import json
import pandas as pd


class ModelReportMetric(object):
    def process(self, input, par_dict=None):
        par_dict = par_dict or {}
        compare_job_list = par_dict.get('job_list')
        job_dir = par_dict.get('job_dir', 'job_result')
        if not compare_job_list:
            raise ValueError('ModelReportMetric: 请在 par_dict 配置 job_list。')

        base = ['AUC', 'precision_score', 'recall_score', 'accuracy_score']
        title = (['job_id']
                 + ['train_' + item for item in base]
                 + ['test_' + item for item in base]
                 + ['model_pars', 'train_num', 'test_num', 'feature_names', 'input_dir'])

        all_record_list = []
        for job_id in compare_job_list:
            in_file = job_dir + '/' + job_id + '/model_result'
            job_info = json.load(open(in_file))
            ti = job_info.get('train_info', job_info.get('train', {}))
            record = ([job_id]
                      + [job_info['train'][item] for item in base]
                      + [job_info['test'][item] for item in base]
                      + [job_info['pars'],
                         ti.get('train_num', job_info.get('train_num')),
                         ti.get('test_num', job_info.get('test_num')),
                         ti.get('feature_names', job_info.get('feature_names')),
                         job_info.get('input_dir', '')])
            all_record_list.append(record)

        compare_df = pd.DataFrame(all_record_list, columns=title)
        out_path = par_dict.get('out_path')
        if out_path:
            compare_df.T.to_excel(out_path)
        return {'model_compare': compare_df}
