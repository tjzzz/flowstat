#encoding=utf8
"""A/B 实验常用统计工具（基于 numpy / scipy，通用、无业务依赖）。

覆盖：SRM 检验、Welch t 检验、两比例 z 检验、效应量、最小样本量估计、
CUPED 方差缩减、BH 多重比较校正。
"""
import numpy as np
from scipy import stats


def srm_test(counts, expected=None):
    """Sample Ratio Mismatch 检验（卡方拟合优度）。

    counts:   各组实际样本量 list/array
    expected: 各组期望比例(list, 和=1)，默认均分
    返回: chi2, dof, p_value, expected_counts, pass(p>0.01 视为通过)
    """
    counts = np.asarray(counts, dtype=float)
    k = len(counts)
    if expected is None:
        expected = np.ones(k) / k
    expected = np.asarray(expected, dtype=float)
    expected = expected / expected.sum()
    exp_counts = counts.sum() * expected
    chi2, p = stats.chisquare(f_obs=counts, f_exp=exp_counts)
    return {
        'chi2': float(chi2),
        'dof': int(k - 1),
        'p_value': float(p),
        'expected_counts': exp_counts.tolist(),
        'pass': bool(p > 0.01),
    }


def welch_ttest(ctrl, treat, alpha=0.05, two_sided=True):
    """Welch 两样本 t 检验（不假设方差齐性，A/B 默认）。

    ctrl, treat: 数值数组
    返回: 均值、差值、绝对/相对提升、t 统计量、p 值、差值置信区间、Cohen's d
    """
    ctrl = np.asarray(ctrl, dtype=float)
    treat = np.asarray(treat, dtype=float)
    m_c, m_t = float(ctrl.mean()), float(treat.mean())
    diff = m_t - m_c
    rel = diff / m_c if m_c != 0 else float('nan')
    alt = 'two-sided' if two_sided else 'greater'
    t_stat, p = stats.ttest_ind(treat, ctrl, equal_var=False, alternative=alt)
    n_c, n_t = len(ctrl), len(treat)
    se = np.sqrt(ctrl.var(ddof=1) / n_c + treat.var(ddof=1) / n_t)
    z = stats.norm.ppf(1 - alpha / 2) if two_sided else stats.norm.ppf(1 - alpha)
    return {
        'mean_ctrl': m_c, 'mean_treat': m_t,
        'diff': diff, 'abs_lift': diff, 'rel_lift': rel,
        't_stat': float(t_stat), 'p_value': float(p),
        'ci_low': float(diff - z * se), 'ci_high': float(diff + z * se),
        'cohens_d': cohens_d(ctrl, treat),
    }


def two_proportion_test(x_ctrl, n_ctrl, x_treat, n_treat, alpha=0.05, two_sided=True):
    """两比例 z 检验（转化率 / 留存率等）。

    x: 转化数, n: 样本数
    返回: 比例、差值、绝对/相对提升、z 统计量、p 值、差值置信区间
    """
    p_c = x_ctrl / n_ctrl
    p_t = x_treat / n_treat
    diff = p_t - p_c
    rel = diff / p_c if p_c != 0 else float('nan')
    pool = (x_ctrl + x_treat) / (n_ctrl + n_treat)
    se = np.sqrt(pool * (1 - pool) * (1 / n_ctrl + 1 / n_treat))
    z = diff / se if se > 0 else 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z))) if two_sided else (1 - stats.norm.cdf(abs(z)))
    se_u = np.sqrt(p_c * (1 - p_c) / n_ctrl + p_t * (1 - p_t) / n_treat)
    zc = stats.norm.ppf(1 - alpha / 2) if two_sided else stats.norm.ppf(1 - alpha)
    return {
        'p_ctrl': float(p_c), 'p_treat': float(p_t),
        'diff': float(diff), 'abs_lift': float(diff), 'rel_lift': rel,
        'z_stat': float(z), 'p_value': float(p),
        'ci_low': float(diff - zc * se_u), 'ci_high': float(diff + zc * se_u),
    }


def cohens_d(ctrl, treat):
    """Cohen's d 效应量。"""
    ctrl = np.asarray(ctrl, dtype=float)
    treat = np.asarray(treat, dtype=float)
    n_c, n_t = len(ctrl), len(treat)
    sp = np.sqrt(((n_c - 1) * ctrl.var(ddof=1) + (n_t - 1) * treat.var(ddof=1)) / (n_c + n_t - 2))
    return 0.0 if sp == 0 else float((treat.mean() - ctrl.mean()) / sp)


def min_sample_size_mean(cv, delta, alpha=0.05, power=0.8, two_sided=True, relative=True):
    """连续指标最小样本量（每组）。

    cv:    目标指标变异系数 = std/mean
    delta: 预期提升(MDE)；relative=True 时为相对提升(0.05=+5%)，False 时为绝对差
    返回: n_per_group, total_n, z_alpha, z_beta, note
    """
    z_a = stats.norm.ppf(1 - alpha / 2) if two_sided else stats.norm.ppf(1 - alpha)
    z_b = stats.norm.ppf(power)
    ratio = cv / delta if delta != 0 else float('inf')
    n = 2 * (z_a + z_b) ** 2 * ratio ** 2
    return {
        'n_per_group': int(np.ceil(n)),
        'total_n': int(np.ceil(2 * n)),
        'z_alpha': float(z_a), 'z_beta': float(z_b),
        'note': 'relative MDE' if relative else 'absolute MDE',
    }


def min_sample_size_proportion(p_baseline, mde, alpha=0.05, power=0.8, two_sided=True):
    """比例指标最小样本量（每组）。

    p_baseline: 基线转化率；mde: 绝对最小可检测差(0.02=+2pp)
    返回: n_per_group, total_n
    """
    z_a = stats.norm.ppf(1 - alpha / 2) if two_sided else stats.norm.ppf(1 - alpha)
    z_b = stats.norm.ppf(power)
    p1, p2 = p_baseline, p_baseline + mde
    pbar = (p1 + p2) / 2
    n = ((z_a * np.sqrt(2 * pbar * (1 - pbar)) + z_b * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2) / (mde ** 2)
    return {'n_per_group': int(np.ceil(n)), 'total_n': int(np.ceil(2 * n))}


def cuped_adjust(y, x, treat_flag):
    """CUPED 方差缩减：用实验前协变量 x 对实验期指标 y 做调整。

    y: 实验期指标数组; x: 协变量数组(长度同 y); treat_flag: 0/1 数组
    返回: theta, 方差缩减百分比, 调整后均值(ctrl/treat), 调整后差值
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    treat_flag = np.asarray(treat_flag, dtype=float)
    cov = np.cov(x, y)
    theta = cov[0, 1] / cov[0, 0] if cov[0, 0] != 0 else 0.0
    y_adj = y - theta * (x - x.mean())
    m_c = y_adj[treat_flag == 0].mean()
    m_t = y_adj[treat_flag == 1].mean()
    var_raw = y.var(ddof=1)
    var_adj = y_adj.var(ddof=1)
    reduction = (1 - var_adj / var_raw) * 100 if var_raw != 0 else 0.0
    return {
        'theta': float(theta),
        'var_reduction_pct': float(reduction),
        'mean_ctrl_adj': float(m_c), 'mean_treat_adj': float(m_t),
        'diff_adj': float(m_t - m_c),
    }


def benjamini_hochberg(p_values, alpha=0.05):
    """BH 多重比较校正（多指标同时检验时控制 FDR）。

    p_values: list
    返回: 原 p、校正后 p、是否拒绝(bool list)
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    bh = ranked * m / np.arange(1, m + 1)
    bh = np.minimum.accumulate(bh[::-1])[::-1]
    bh = np.clip(bh, 0, 1)
    rej = bh <= alpha
    adj = np.empty(m, dtype=float)
    adj[order] = bh
    out = np.empty(m, dtype=bool)
    out[order] = rej
    return {'p_values': p.tolist(), 'adjusted': adj.tolist(), 'rejected': out.tolist()}
