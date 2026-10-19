#encoding=utf8
"""
FlowStat 总入口（已脱敏，去除原 aibee 凭证/内网/域名）。

用法:
    python run.py -e examples/1_read_data.yaml -m demo1_read_data
或在代码中:
    from run import run
    run('examples/1_read_data.yaml', 'demo1_read_data', {})
"""
import sys
import os
import logging
logging.basicConfig(level=logging.INFO)
import argparse
import time

run_script_path = os.path.split(os.path.realpath(__file__))[0] + '/'
sys.path.append(run_script_path)

from readers.reader_local import get_all_readers as get_local_readers
from utils import ExpConf, StreamTask, check_table_exists, get_split_job_list
from src.send_mail import send_mail


def run_task(df_dict, exp_conf, task):
    logging.info('start to run ....... day=' + df_dict.get('run_day', '-'))

    # 1) 本地 reader（基于 pandas，无需 spark）
    if len(exp_conf.reader_list) > 0:
        local_df, local_names = get_local_readers(exp_conf.reader_list, df_dict)
        df_dict.update(local_df)
        df_dict.update({'local_df_name_list': local_names})

    df_out = task.process(df_dict, exp_conf.metric_pars_dict)
    return df_out, 0


def run(exp_yaml_file, m_stream, user_input_dict):
    df_dict = {}
    exp_conf = ExpConf(exp_yaml_file, m_stream)
    df_dict.update(exp_conf.conf)
    df_dict.update(exp_conf.common_metric_pars_dict)
    df_dict.update(user_input_dict)
    df_dict['FLOWSTAT_DIR'] = run_script_path

    if 'brand_city_site' not in df_dict and 'brand' in df_dict:
        df_dict['brand_city_site'] = df_dict['brand'] + '_' + df_dict['city'] + '_' + df_dict['site']

    if 'run_day' not in df_dict:
        start_dt = df_dict.get('start_dt', 'day1')
        end_dt = df_dict.get('end_dt', 'day2')
        df_dict['run_day'] = start_dt if start_dt == end_dt else start_dt + '_' + end_dt

    check_res = check_table_exists(
        df_dict.get('brand_city_site', ''), df_dict.get('run_day', ''))
    if check_res != 0:
        print('log not ready')
        send_mail(df_dict.get('brand_city_site', ''), df_dict.get('run_day', '') + ' fail')
        return

    task = StreamTask(exp_conf.metric_dir_list, exp_conf.metric_list)

    if int(df_dict.get('run_days', 0)) > 0:
        start_dt = df_dict.get('start_dt')
        end_dt = df_dict.get('end_dt')
        job_list = get_split_job_list(start_dt, end_dt)
        print('run_串行 tasks.....')
        for st, et in job_list:
            df_dict['start_dt'] = st
            df_dict['end_dt'] = et
            df_dict['run_day'] = st + '_' + et
            df_out, _ = run_task(df_dict, exp_conf, task)
    else:
        df_out, _ = run_task(df_dict, exp_conf, task)

    return df_out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", help="实验配置文件 exp.yaml", required=True)
    parser.add_argument("-m", help="要运行的 stream 名", required=True)
    parser.add_argument("-run_day", help="运行日期", type=str)
    parser.add_argument("-brand", help="brand", type=str)
    parser.add_argument("-city", help="city", type=str)
    parser.add_argument("-site", help="site", type=str)
    parser.add_argument("-set", help="其他参数 -set k1:v1,k2:v2")
    parser.add_argument("-run_days", help="是否按天串联跑(1开启)", type=int, default=0)
    args = parser.parse_args()
    user_input_dict = {}
    tmp_info = vars(args)
    for key in tmp_info:
        if tmp_info[key] is not None:
            if key == 'set':
                for kv in tmp_info['set'].split(','):
                    k, v = kv.split(':', 1)
                    user_input_dict[k] = v
            else:
                user_input_dict[key] = tmp_info[key]
    return user_input_dict


if __name__ == '__main__':
    t0 = time.time()
    user_input_dict = parse_args()
    exp_yaml_file = user_input_dict['e']
    m_stream = user_input_dict['m']
    df_out = run(exp_yaml_file, m_stream, user_input_dict)
    logging.info('run time ' + str(time.time() - t0))
