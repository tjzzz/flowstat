#encoding=utf8
"""最小样本量规划 metric：连续指标(基于 CV + 相对提升) 或 比例指标(基线 + 绝对 MDE)。

无需输入数据，按参数直接估算每组所需样本量。
"""
from src.abtools import min_sample_size_mean, min_sample_size_proportion


class SampleSizeMetric(object):
    def process(self, input, par_dict=None):
        par_dict = par_dict or {}
        alpha = float(par_dict.get('alpha', 0.05))
        power = float(par_dict.get('power', 0.8))
        mode = par_dict.get('mode', 'mean')  # 'mean' | 'proportion'
        out = {}

        if mode == 'mean':
            cv = float(par_dict['cv'])
            delta = float(par_dict['delta'])
            r = min_sample_size_mean(
                cv, delta, alpha, power,
                two_sided=par_dict.get('two_sided', True),
                relative=par_dict.get('relative', True))
            out['sample_size_mean'] = r
            print('[sample_size|mean] n/group=%d total=%d  (cv=%.3f delta=%.3f power=%.2f)'
                  % (r['n_per_group'], r['total_n'], cv, delta, power))
        else:
            p = float(par_dict['p_baseline'])
            mde = float(par_dict['mde'])
            r = min_sample_size_proportion(p, mde, alpha, power)
            out['sample_size_proportion'] = r
            print('[sample_size|proportion] n/group=%d total=%d  (p=%.3f mde=%.3f power=%.2f)'
                  % (r['n_per_group'], r['total_n'], p, mde, power))
        return out
