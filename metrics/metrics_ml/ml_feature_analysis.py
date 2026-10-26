#encoding=utf8
"""特征分析 metric：自动检测特征间显著性关系（卡方 / 方差分析 / 皮尔逊 / 序列）。

移植自 aibee logstat 的 metrics/models/feature_analysis.py，已脱敏（去除内部报告组件）。
返回各维度检验结论；可配置 report_path 用 pandas 导出 Excel。
"""
import logging
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, pearsonr

from metrics.metrics_ml.ml_utils import get_input_data


def get_pair_list(target, feature_name_list, stage=1):
    if target != '-':
        return [target + '@' + feat for feat in feature_name_list]
    pair_list = []
    n = len(feature_name_list)
    for i in range(n):
        for j in range(i + 1, n):
            pair_list.append(feature_name_list[i] + '@' + feature_name_list[j])
    return pair_list


def get_name_list(par_dict, type_name):
    class_name = par_dict.get(type_name, '-')
    if class_name == '-':
        return []
    return class_name.split('@')


def trans_multi(df, multi_class_name, label='pid'):
    return pd.DataFrame({
        label: df[label].repeat(df[multi_class_name].str.len()),
        multi_class_name: np.concatenate(df[multi_class_name].values),
    })


class CorrelationAnalysis(object):
    """自动检测不同特征变量之间的显著性关系。"""

    def __init__(self):
        self.ALPHA = 0.01

    def explore_metric_features(self, mm1, mm2, df):
        df2 = df[[mm1, mm2]].dropna()
        lo1, up1 = np.percentile(df[mm1], [2, 98])
        lo2, up2 = np.percentile(df[mm1], [2, 98])
        df2 = df2[(df2[mm1] > lo1) & (df2[mm2] <= up1)]
        df2 = df2[(df2[mm2] > lo2) & (df2[mm2] <= up2)]
        r, p_value = pearsonr(df2[mm1], df2[mm2])
        return round(p_value, 3), r

    def explore_category_features(self, mm1, mm_list, df):
        from statsmodels.formula.api import ols
        from statsmodels.stats.anova import anova_lm
        df2 = df[[mm1] + mm_list].dropna()
        formula = mm1 + ' ~ C(' + ') + C('.join(mm_list) + ')'
        anova_results = anova_lm(ols(formula, df2).fit())
        p_value = anova_results['PR(>F)'][0]
        res = df2.groupby(mm_list).agg({mm1: ['mean', 'median', 'count']}).reset_index()
        return round(p_value, 3), res

    def chi2_test(self, mm1, mm2, df):
        df2 = df[[mm1, mm2]].dropna()
        count_table = pd.crosstab(df2[mm1], df2[mm2])
        chi2, p, dof, ex = chi2_contingency(count_table)
        return round(p, 3), count_table, chi2

    def extract_informative_sequence(self, category_name, sequence_name, df):
        act_list_all = list(df[sequence_name])
        window = 3
        base_item_dict, id_act_dict, count_dict = {}, {}, {}
        index = 0
        for i, row in df.iterrows():
            act_list = eval(row[sequence_name])
            act_len = len(act_list)
            if act_len >= window:
                for j in range(act_len - window + 1):
                    act_item = '@'.join(act_list[j: j + window])
                    if act_item not in base_item_dict:
                        index += 1
                        base_item_dict[act_item] = index
                        id_act_dict[index] = act_item
                        count_dict[act_item] = 0
                    count_dict[act_item] += 1
            else:
                act_item = '@'.join(act_list)
                if act_item not in base_item_dict:
                    index += 1
                    base_item_dict[act_item] = index
                    id_act_dict[index] = act_item
                    count_dict[act_item] = 0
                count_dict[act_item] += 1
        top_act_item_list = [k for k, v in count_dict.items() if v > 10]
        ncol = len(top_act_item_list)
        group_value_list, mm_matrix = [], []
        for i, row in df.iterrows():
            vector = np.zeros(ncol)
            act_list = eval(row[sequence_name])
            act_len = len(act_list)
            if act_len >= window:
                for j in range(act_len - window + 1):
                    act_item = '@'.join(act_list[j: j + window])
                    if act_item in top_act_item_list:
                        vector[top_act_item_list.index(act_item)] = 1
            else:
                act_item = '@'.join(act_list)
                if act_item in top_act_item_list:
                    vector[top_act_item_list.index(act_item)] = 1
            group_value_list.append(df[category_name])
            mm_matrix.append(vector)
        final_df = pd.DataFrame(mm_matrix, columns=top_act_item_list)
        final_df[category_name] = df[category_name].values
        significant_chi2_list = []
        for col in top_act_item_list:
            count_table = pd.crosstab(final_df[col], final_df[category_name])
            chi2, p, dof, ex = chi2_contingency(count_table)
            if p < 0.05:
                significant_chi2_list.append([chi2, col, count_table])
        significant_act, significant_count_table = '', ''
        if significant_chi2_list:
            significant_chi2_list.sort(reverse=True)
            significant_chi2, significant_act, significant_count_table = significant_chi2_list[0]
        return significant_act, significant_count_table

    def exploreAll(self, df, target, class_name_list, continuous_list, sequence_list, show_plot=0):
        all_check_result = [[], [], []]
        pair_list = get_pair_list(target, class_name_list)
        for pair in pair_list:
            name1, name2 = pair.split('@')
            p_value, count_table, chi2 = self.chi2_test(name1, name2, df)
            count_table_rate = count_table.div(count_table.sum(axis=1), axis=0).T
            conclusion = '两者无显著关系'
            if p_value < self.ALPHA:
                conclusion = '两者非独立(chi2=%s, p=%s)' % (round(chi2, 0), p_value)
            record = {'title': name1 + ' vs ' + name2, 'conclusion': conclusion, 'data': count_table_rate}
            all_check_result[0].append(record)

        if target == '-':
            if len(continuous_list) >= 2:
                data = df[continuous_list]
                corr_matrix = data.corr()
                for i in range(len(continuous_list)):
                    for j in range(i + 1, len(continuous_list)):
                        name1, name2 = continuous_list[i], continuous_list[j]
                        r = corr_matrix[name1][name2]
                        if abs(r) > 0.6:
                            p_value, rr = self.explore_metric_features(name1, name2, df)
                            if p_value < self.ALPHA:
                                all_check_result[1].append(
                                    {'title': name1 + ' vs ' + name2, 'conclusion': str(r), 'data': r})
        else:
            pair_list = get_pair_list(target, continuous_list)
            for pair in pair_list:
                name1, name2 = pair.split('@')
                p_value, label_mean_value = self.explore_category_features(name2, [name1], df)
                all_check_result[2].append({
                    'title': 'category*continuous=' + name1 + ' ' + name2,
                    'conclusion': '不同类别均值比较 p=%s' % p_value, 'data': label_mean_value})

        pair_list = get_pair_list(target, sequence_list)
        for pair in pair_list:
            name1, name2 = pair.split('@')
            significant_act, significant_count_table = self.extract_informative_sequence(name1, name2, df)
            if significant_act != '':
                for mm in list(set(df[name1])):
                    num_0 = significant_count_table[mm][0]
                    num_1 = significant_count_table[mm][1]
                    print(mm, str(round(num_1 / (num_1 + num_0), 4) * 100) + '%')
        return all_check_result


class FeatureAnalysisMetric(object):
    def process(self, input, par_dict=None):
        par_dict = par_dict or {}
        logging.basicConfig(level=logging.ERROR)

        data = get_input_data(input, par_dict)
        if data is None:
            raise ValueError('FeatureAnalysisMetric: 未找到分析数据。')

        target = par_dict.get('label', '-')
        show_plot = int(par_dict.get('show_plot', 0))
        class_name_list = get_name_list(par_dict, 'class')
        continuous_list = get_name_list(par_dict, 'continuous')
        sequence_list = get_name_list(par_dict, 'sequence')

        ana = CorrelationAnalysis()
        all_check_result = ana.exploreAll(data, target, class_name_list, continuous_list, sequence_list, show_plot)

        multi_class_name_list = get_name_list(par_dict, 'multi_class')
        for multi_name in multi_class_name_list:
            tmp_data = data[[target, multi_name]].copy()
            tmp_data[multi_name] = tmp_data[multi_name].map(lambda x: eval(x))
            data_trans = trans_multi(tmp_data, multi_name, target)
            tmp_result = ana.exploreAll(data_trans, target, [multi_name], [], [], show_plot)
            all_check_result[0] += tmp_result[0]

        report_path = par_dict.get('report_path')
        if report_path:
            with pd.ExcelWriter(report_path) as writer:
                sheet_names = ['categorical', 'continuous', 'category_x_continuous']
                for records, sheet in zip(all_check_result, sheet_names):
                    rows = [{'title': r['title'], 'conclusion': r['conclusion']} for r in records]
                    pd.DataFrame(rows).to_excel(writer, sheet_name=sheet, index=False)

        return {'feature_analysis': all_check_result}
