"""ML 工具函数：数据读取、文件/字符串 md5、job id、工作日判断。

移植自 aibee logstat 的 metrics/models/utils_ml.py，已脱敏。
`get_input_data` 同时兼容 FlowStat 的 df_dict（从 reader 拿到的 DataFrame）与外部 CSV 文件。
"""
import os
import time
import hashlib
import datetime

import pandas as pd


def get_input_data(input, par_dict, sep='\t'):
    """读取建模数据。优先级：df_dict 中已存在的 DataFrame > 外部文件。

    - FlowStat 场景：reader 产出的 DataFrame 放在 input[df_key] 或 input['df_features']
    - 独立场景：通过 par_dict['data_dir'] / input['data_dir'] 指定 CSV/TSV 路径
    """
    df_key = par_dict.get('df_key', input.get('df_name', None))
    if df_key and isinstance(input.get(df_key), pd.DataFrame):
        return input[df_key]
    if isinstance(input.get('df_features'), pd.DataFrame):
        return input['df_features']

    data_dir = par_dict.get('data_dir', input.get('data_dir', '-'))
    if data_dir == '-':
        return None
    return pd.read_csv(data_dir, sep=sep)


def get_is_workday(dt):
    """判断某天是否为工作日（中国法定，剔除法定节假日，含调休补班）。"""
    from chinese_calendar import is_workday as _is_workday  # 懒加载：仅用到时才需要该依赖
    if '-' in dt:
        year, month, day = dt.split('-')
    else:
        year = dt[:4]
        month = dt[4:6]
        day = dt[6:]
    da = datetime.date(int(year), int(month), int(day))
    return int(_is_workday(da))


def sigmoid(x):
    x = int(x)
    return 1 / (1 + pow(2.718281828, -0.01 * (x - 300))) - 1 / (1 + pow(2.718281828, 3))


def get_file_md5(file_name):
    """计算文件 md5。"""
    m = hashlib.md5()
    with open(file_name, 'rb') as fobj:
        while True:
            data = fobj.read(4096)
            if not data:
                break
            m.update(data)
    return m.hexdigest()


def get_str_md5(content):
    """计算字符串 md5。"""
    m = hashlib.md5(content)
    return m.hexdigest()


def get_job_id():
    """为每个 job 分配基于时间的唯一 id。"""
    return time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())


def perform_data_trans(df_features, label_col, categorical_features_index, feature_names, out_dir):
    """对类别特征做 LabelEncoder 编码，并持久化映射表便于预测时复用。"""
    from sklearn.preprocessing import LabelEncoder

    code_mapping_dict = {}
    lbe = LabelEncoder()

    if os.path.exists(out_dir + '/save.code_mapping_dict'):
        for line in open(out_dir + '/save.code_mapping_dict'):
            dim, attr, index = line.strip().split('\t')
            if dim not in code_mapping_dict:
                code_mapping_dict[dim] = {}
            if attr not in code_mapping_dict[dim]:
                code_mapping_dict[dim][attr] = int(index)

        def _trans(attr, mapping_dict, feat):
            max_num = len(mapping_dict[feat])
            return mapping_dict[feat].get(str(attr), max_num)

        for i in categorical_features_index:
            feat = feature_names[i]
            df_features[feat] = df_features[feat].apply(_trans, **{'mapping_dict': code_mapping_dict, 'feat': feat})
    else:
        ff = open(out_dir + '/save.code_mapping_dict', 'w')
        for i in categorical_features_index:
            feat = feature_names[i]
            df_features[feat] = lbe.fit_transform(df_features[feat])
            zz = list(lbe.classes_)
            code_mapping_dict[feat] = dict(zip(zz, range(len(zz))))
            for j in range(len(zz)):
                ff.write('\t'.join(map(str, [feat, zz[j], j])) + '\n')
        df_features[label_col] = lbe.fit_transform(df_features[label_col])
        zz = list(lbe.classes_)
        code_mapping_dict[label_col] = dict(zip(zz, range(len(zz))))
        for j in range(len(zz)):
            ff.write('\t'.join(map(str, [label_col, zz[j], j])) + '\n')
        ff.close()

    return df_features, code_mapping_dict


def get_train_test_data(df_features, input, label_col):
    """按配置切分训练/测试集：比例切分 或 按时间切分，可选类别均衡采样。"""
    train_test_method = input['train_test_method']
    if '0' in train_test_method and '-' in train_test_method:   # 0.8-0.2
        p1, p2 = train_test_method.split('-')
        from sklearn.model_selection import train_test_split
        p2 = float(p2)
        train, test = train_test_split(df_features, test_size=p2)

    if train_test_method == 'dt':   # 按时间切分
        train_start = input['train_start']
        train_end = input['train_end']
        train = df_features[(df_features['dt'] >= train_start) & (df_features['dt'] <= train_end)]
        test = df_features[df_features['dt'] > train_end]

    if int(input.get('balance_data', 0)) == 1:   # 人工上采样少数类
        train1 = train[train[label_col] == 1]
        train2 = train[train[label_col] == 0]
        train = pd.concat([train1, train2.sample(train1.shape[0])])

    return train, test
