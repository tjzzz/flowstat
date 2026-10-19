"""深度更新嵌套字典：u 的键值递归覆盖 d（用于 metric 输出合并到共享 df_dict）。"""


try:
    from collections.abc import Mapping
except ImportError:  # pragma: no cover
    from collections import Mapping


def recursive_update(d, u):
    for k, v in u.items():
        if isinstance(v, Mapping):
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d
