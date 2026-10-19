#encoding=utf8
"""A/B 实验分析 metric：SRM 检验 + 均值(t 检验)/比例(z 检验) + 提升与置信区间。

依赖 src.abtools。输入 DataFrame 需含分组列(如 group)、数值指标列(如 value)，
可选转化率列(如 converted, 0/1)。
"""
import numpy as np
from src.abtools import srm_test, welch_ttest, two_proportion_test


class ABTestMetric(object):
    def process(self, input, par_dict=None):
        par_dict = par_dict or {}
        df_name = par_dict.get('df', 'ab')
        df = input[df_name]
        group_col = par_dict.get('group_col', 'group')
        value_col = par_dict.get('value_col', 'value')
        conv_col = par_dict.get('conv_col', None)
        alpha = float(par_dict.get('alpha', 0.05))

        groups = sorted([str(g) for g in df[group_col].unique()])
        if len(groups) != 2:
            raise ValueError('ABTestMetric 仅支持两组实验，当前分组: %s' % groups)
        ctrl_name, treat_name = groups[0], groups[1]
        ctrl = df[df[group_col].astype(str) == ctrl_name]
        treat = df[df[group_col].astype(str) == treat_name]
        n_c, n_t = len(ctrl), len(treat)

        srm = srm_test([n_c, n_t])

        out = {
            'ab': {
                'groups': {ctrl_name: n_c, treat_name: n_t},
                'srm': srm,
            }
        }

        t = welch_ttest(ctrl[value_col].values, treat[value_col].values, alpha=alpha)
        out['ab']['value_test'] = t

        if conv_col:
            x_c, x_t = int(ctrl[conv_col].sum()), int(treat[conv_col].sum())
            z = two_proportion_test(x_c, n_c, x_t, n_t, alpha=alpha)
            out['ab']['conv_test'] = z

        # 摘要
        print('[SRM] p=%.4f -> %s' % (srm['p_value'], 'PASS' if srm['pass'] else 'FAIL'))
        print('[value] %s=%.4f %s=%.4f | diff=%.4f rel_lift=%.2f%% | p=%.4f CI=[%.4f,%.4f]'
              % (ctrl_name, t['mean_ctrl'], treat_name, t['mean_treat'], t['diff'],
                 t['rel_lift'] * 100, t['p_value'], t['ci_low'], t['ci_high']))
        if conv_col:
            print('[conv] %s=%.4f %s=%.4f | diff=%.4f rel_lift=%.2f%% | p=%.4f'
                  % (ctrl_name, z['p_ctrl'], treat_name, z['p_treat'],
                     z['diff'], z['rel_lift'] * 100, z['p_value']))
        return out
