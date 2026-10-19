#encoding=utf8
"""
FlowStat 实验配置与任务调度工具（通用）。

- ExpConf:    解析 exp.yaml（conf 基础配置 + 指定 stream 下的 metric 列表）
- StreamTask: 按类名扫描 metric 目录并链式加载实例，依次 process 并深度合并输出
"""
import yaml
import sys
import os
import inspect
import logging
import math

PWD = os.path.split(os.path.realpath(__file__))[0] + '/'
sys.path.append(PWD + 'readers/')
sys.path.append(PWD + 'src/')
sys.path.append(PWD + 'metrics/')

from dict_recursive_update import recursive_update
from src.tools import get_days


def get_split_job_list(start_dt, end_dt):
    """将任务按天拆分成多段，便于串行/并行执行。"""
    PARA_NUM = 7
    day_list = get_days(start_dt, end_dt)
    if len(day_list) <= PARA_NUM:
        return [[start_dt, end_dt]]
    num = math.ceil(len(day_list) / PARA_NUM)
    job_list = []
    for i in range(num):
        if i < num - 1:
            job_list.append([day_list[i * PARA_NUM], day_list[(i + 1) * PARA_NUM - 1]])
        else:
            job_list.append([day_list[i * PARA_NUM], day_list[-1]])
    return job_list


def check_table_exists(brand_city_site, dt):
    """
    数据就绪检查（占位）。
    早期版本会 ssh 到 HDFS 检查分区；通用版请在此接入你自己的存储（对象存储/数据库）的就绪探测。
    返回 0 表示直接放行。
    """
    return 0


def get_class_name(module_name, metric_list):
    """在 module 中查找 metric_list 里声明的类。"""
    module = __import__(module_name)
    class_name = '-'
    for tmp in inspect.getmembers(module, inspect.isclass):
        if tmp[0] in metric_list:
            class_name = tmp[1]
    return class_name


class ExpConf(object):
    """解析 exp.yaml：conf 基础配置 + 指定 stream 下的 metric 列表。"""

    def __init__(self, exp_conf, m_stream):
        exp_info = yaml.load(open(exp_conf), Loader=yaml.FullLoader)
        self.common_metric_pars_dict = {}
        self.reader_list = []

        self.conf = exp_info.get('conf', {})
        if 'conf' in exp_info:
            tmp_metric_dir_list = exp_info['conf']['metric_dir'].split(',')
            metric_dir_list = [PWD + 'metrics/tools/']
            for metric_dir in tmp_metric_dir_list:
                metric_dir = metric_dir.strip()
                if metric_dir.startswith('metrics/'):
                    metric_dir_list.append(PWD + metric_dir)
                else:
                    metric_dir_list.append(metric_dir)
            self.metric_dir_list = metric_dir_list

        metric_list = []
        metric_pars_dict = []
        for mm in exp_info[m_stream]:
            if isinstance(mm, dict):
                metric = list(mm.keys())[0]
                par_str = mm[metric]
                if isinstance(par_str, dict):
                    if metric == 'Common':
                        self.common_metric_pars_dict = par_str
                    else:
                        metric_list.append(metric)
                        metric_pars_dict.append(par_str)
                else:
                    par_str = par_str.replace(' ', '')
                    if metric == 'reader':
                        self.reader_list = par_str.split(',')
                    else:
                        par_dict = {}
                        for item in par_str.split(','):
                            if not item:
                                continue
                            k, v = item.split(':', 1)
                            par_dict[k] = v
                        if metric == 'Common':
                            self.common_metric_pars_dict = par_dict
                        else:
                            metric_pars_dict.append(par_dict)
                            metric_list.append(metric)
            else:
                metric_list.append(mm)
                metric_pars_dict.append({})

        self.metric_list = metric_list
        self.metric_pars_dict = metric_pars_dict


class StreamTask(object):
    """控制任务运行流程：加载 metric 实例并链式 process。"""

    def __init__(self, metric_dir_list, metric_list):
        metric_instance_name_list = self.import_metrics(metric_dir_list, metric_list)
        self.metric_instance_name_list = metric_instance_name_list
        self.metric_list = metric_list

    def import_metrics(self, metric_dir_list, metric_list):
        file_list = []
        for metric_dir in metric_dir_list:
            if not os.path.isdir(metric_dir):
                continue
            file_name_list = os.listdir(metric_dir + '/')
            sys.path.append(metric_dir + '/')
            for file_name in file_name_list:
                if file_name.endswith('.py'):
                    file_list.append(file_name[:-3])

        metric_instance_name_list = []
        class_instance_dict = {}
        for module_name in file_list:
            class_ = get_class_name(module_name, metric_list)
            if class_ == '-':
                continue
            class_name = class_.__name__
            if class_name in metric_list or module_name in metric_list:
                module = __import__(module_name)
                class_ = getattr(module, class_name)
                class_instance_dict[class_name] = class_
        for class_name in metric_list:
            instance = class_instance_dict[class_name]()
            metric_instance_name_list.append([class_name, instance])
        return metric_instance_name_list

    def process(self, df_dict, metric_pars_dict):
        out_dict = df_dict.copy()
        for i in range(len(self.metric_list)):
            metric = self.metric_list[i]
            metric_instance = self.metric_instance_name_list[i][1]
            logging.info('====In process... Metric = ' + metric)
            metric_pars = metric_pars_dict[i]
            if len(metric_pars) > 0:
                output = metric_instance.process(out_dict, metric_pars)
            else:
                output = metric_instance.process(out_dict)
            if output is not None:
                out_dict = recursive_update(out_dict, output)
        return out_dict
