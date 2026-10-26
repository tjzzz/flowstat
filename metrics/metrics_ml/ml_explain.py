#encoding=utf8
"""模型可解释性 metric：SHAP 特征归因 + 按维度拆解定位预测错误（卡方/ t 检验）。

移植自 aibee logstat 的 metrics/models/model_explain.py，已脱敏（去除内部报告组件）。
返回 SHAP 值与逐维度分析结论，由调用方决定如何落盘/展示。
"""
import os
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency


def prepare_case_data(case_data, model, eva_label):
    debug_data = case_data.values
    pre_score = list(model.predict(debug_data))
    case_data = case_data.copy()
    case_data['pre'] = [int(float(score) > 0.5) for score in pre_score]
    case_data['label'] = eva_label
    case_data['pre_right'] = (case_data['pre'] == case_data['label']).map(int)
    case_data['com'] = case_data['label'].map(str) + '@' + case_data['pre'].map(str)
    return case_data


def explore_feature_dims(case_data, sparse_features, dense_features, code_mapping_dict=None, plot_flag=0):
    res_list, res_list2 = [], []
    code_mapping_dict = code_mapping_dict or {}

    def _inv(new_code, inv_dict):
        return inv_dict.get(new_code, new_code)

    if code_mapping_dict:
        for cat_feat in sparse_features:
            inv_dict = {v: k for k, v in code_mapping_dict.get(cat_feat, {}).items()}
            if inv_dict:
                case_data[cat_feat] = case_data[cat_feat].apply(_inv, **{'inv_dict': inv_dict})

    for cat_feat in sparse_features:
        count_table = pd.crosstab(case_data[cat_feat], case_data['com'])
        chi2, p_value, dof, ex = chi2_contingency(count_table)
        p_value = round(p_value, 3)
        count_table[0] = count_table['0@0'] + count_table['0@1']
        count_table[1] = count_table['1@0'] + count_table['1@1']
        count_table['err_rate'] = count_table[0] / (count_table[0] + count_table[1])
        if p_value < 0.05:
            count_table = count_table.sort_values(by=1, ascending=False)
            bad_case = count_table[count_table['err_rate'] > 0.4]
            res_list.append({'title': cat_feat, 'data': count_table, 'conclusion': 'p=%s' % p_value})
            if plot_flag:
                print(cat_feat, 'p_value=', p_value, '\n', count_table)
                if bad_case.shape[0] > 0:
                    print('err_rate>0.4\n', bad_case)

    for con_feat in dense_features:
        sample0 = case_data[case_data['pre_right'] == 0][con_feat]
        sample1 = case_data[case_data['pre_right'] == 1][con_feat]
        t = stats.ttest_ind(sample0, sample1)
        p_value2 = round(t.pvalue, 3)
        if p_value2 < 0.05:
            df_stat = pd.DataFrame({
                'wrong': [sample0.mean(), sample0.median(), sample0.skew()],
                'right': [sample1.mean(), sample1.median(), sample1.skew()],
            })
            df_stat.index = ['Mean', 'Median', 'Skew']
            res_list2.append({'title': con_feat, 'data': df_stat, 'conclusion': 'p=%s' % p_value2})
            if plot_flag:
                print(con_feat, 'p_value=', p_value2, sample0.mean(), sample1.mean())
    return [res_list, res_list2], case_data


def get_shap_analysis(model, case_data, feature_names, plot_type='summary'):
    import shap
    X_test = case_data[feature_names]
    explainer = shap.TreeExplainer(model)
    tmp = explainer.shap_values(X_test)
    if isinstance(tmp, list) and len(tmp) == 2:   # 二分类
        shap_values = explainer.shap_values(X_test)[1]
        evalue = explainer.expected_value[1]
    else:
        shap_values = tmp
        evalue = explainer.expected_value

    shap_df = pd.DataFrame(shap_values)
    shap_df.index = case_data.index

    g = None
    if plot_type == 'summary':
        g = shap.summary_plot(shap_values, X_test)
    elif plot_type == 'importance':
        g = shap.summary_plot(shap_values, X_test, plot_type='bar')
    elif plot_type == 'force':
        g = shap.force_plot(evalue, shap_values, X_test)
    return case_data, shap_values, shap_df, evalue, g


class MLExplainMetric(object):
    def process(self, input, par_dict=None):
        par_dict = par_dict or {}
        eva_data = input['eva_data']
        model = input['model']
        eva_label = input['eva_label']
        plot_flag = int(par_dict.get('plot', 0))
        job_result_dir = par_dict.get('job_result_dir', 'explain_result/')
        os.makedirs(job_result_dir, exist_ok=True)

        case_data = prepare_case_data(eva_data, model, eva_label)
        sparse_features = par_dict.get('sparse_features', input.get('sparse_features', []))
        dense_features = par_dict.get('dense_features', input.get('dense_features', []))
        feature_names = list(sparse_features) + list(dense_features)
        case_data_new = case_data.copy()

        report_info_list, format_case_data = explore_feature_dims(
            case_data_new, sparse_features, dense_features,
            input.get('code_mapping_dict', {}), plot_flag)

        shap_out = get_shap_analysis(
            model, case_data, feature_names, par_dict.get('plot_type', 'summary'))

        new_case_data, shap_values, shap_df, expected_value, g = shap_out
        shap_df.to_csv(job_result_dir + 'shap_result.csv', sep='\t')
        format_case_data.to_csv(job_result_dir + 'eva_result.csv', sep='\t')

        return {
            'shap_values': shap_values,
            'feature_dims_report': report_info_list,
            'g': g,
        }
