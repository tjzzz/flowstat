#encoding=utf8
"""通用工具函数（已脱敏，仅保留与业务无关的公共算法/日期工具）。"""
import os
import math
import datetime


def get_day_diff(day1, day2):
    templ = '%Y-%m-%d' if '-' in day1 else '%Y%m%d'
    d1 = datetime.datetime.strptime(day1, templ).date()
    d2 = datetime.datetime.strptime(day2, templ).date()
    return int((d1 - d2).days)


def get_day_add(day, delta):
    templ = '%Y-%m-%d' if '-' in day else '%Y%m%d'
    d1 = datetime.datetime.strptime(day, templ).date() + datetime.timedelta(days=delta)
    return d1.strftime(templ)


def get_days(day0, day1, delta=1):
    """返回 [day0, day1] 之间按天列举的日期列表。"""
    templ = '%Y-%m-%d' if '-' in day0 else '%Y%m%d'
    d1 = datetime.datetime.strptime(day0, templ).date()
    d2 = datetime.datetime.strptime(day1, templ).date()
    step = datetime.timedelta(days=delta)
    out = []
    tmp = d1
    while tmp <= d2:
        out.append(tmp.strftime(templ))
        tmp += step
    return out


############## 距离 / 几何相关 ##############
def cal_distance(point_a, point_b):
    return ((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2) ** 0.5


def point2LineDistance(point_a, point_b, point_c):
    """点 a 到直线 bc 的距离。"""
    a = cal_distance(point_b, point_c)
    b = cal_distance(point_a, point_c)
    c = cal_distance(point_a, point_b)
    p = (a + b + c) / 2
    tmp = p * (p - a) * (p - b) * (p - c)
    s = 0 if tmp < 0 else (tmp ** 0.5)
    return 2 * s / a


def point2SegDistance(point_p, point_a, point_b):
    """点 p 到线段 ab 的距离。"""
    AP = [point_p[0] - point_a[0], point_p[1] - point_a[1]]
    AB = [point_b[0] - point_a[0], point_b[1] - point_a[1]]
    r = (AP[0] * AB[0] + AP[1] * AB[1]) / (AB[0] ** 2 + AB[1] ** 2)
    if r < 0:
        return cal_distance(point_a, point_p)
    if r > 1:
        return cal_distance(point_b, point_p)
    cx = point_a[0] + r * (point_b[0] - point_a[0])
    cy = point_a[1] + r * (point_b[1] - point_a[1])
    return cal_distance([cx, cy], point_p)


def cal_trace_len(x, y):
    """轨迹折线段总长度（需 shapely）。"""
    from shapely.geometry import LineString
    return LineString(list(zip(x, y))).length


def cal_crowLength(x, y):
    """起终点直线距离。"""
    import numpy as np
    start = np.array([x[0], y[0]])
    end = np.array([x[-1], y[-1]])
    return float(np.sqrt(np.dot(end - start, end - start)))


def get_time(num):
    num = int(num)
    ss = '0' + str(num % 60) if num % 60 < 10 else num % 60
    mm = '0' + str(int(num / 60) % 60) if int(num / 60) % 60 < 10 else int(num / 60) % 60
    hour = '0' + str(int(num / 3600)) if int(num / 3600) < 10 else int(num / 3600)
    return ':'.join(map(str, [hour, mm, ss]))


def transtime(t):
    """hour:min:second -> 整数秒。"""
    t = str(t)
    if t == '-1':
        return -1
    h, m, s = t.split(':')
    return 3600 * int(h) + 60 * int(m) + int(s)


def cal_interval(_list1, _list2):
    """两个区间的重合度系数。"""
    x1, y1 = _list1
    x2, y2 = _list2
    if x1 >= y2 or x2 >= y1:
        return 0
    p1 = min(x1, x2)
    p2 = max(y1, y2)
    return (y2 - x2 + y1 - x1 - (p2 - p1)) / (p2 - p1)


def detect_stay_point(x, y, dt_list):
    """
    停留点检测（片段需时间连续，x/y 为真实坐标）。
    依赖 pyinter，按需安装；未安装时抛异常由调用方处理。
    """
    from pyinter import IntervalSet, closed
    MAX_D = 3
    MIN_T = 30
    n = len(x)
    find_res = []
    for i in range(n):
        dt_k = 0
        j = 0
        for k in range(i + 1, n):
            d = math.sqrt((x[k] - x[i]) ** 2 + (y[k] - x[i]) ** 2)
            if d < MAX_D:
                dt_k = dt_list[k] - dt_list[i]
            else:
                j = k
                break
        if dt_k >= MIN_T:
            find_res.append([i, n - 1 if j == 0 else j])
    intervals = IntervalSet([closed(a, b) for a, b in find_res])
    return [[item.lower_value, item.upper_value] for item in intervals]
