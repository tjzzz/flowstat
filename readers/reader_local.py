#encoding=utf8
"""本地（非 spark）reader 机制（通用）。"""
import inspect
import yaml
import os

PWD = os.path.split(os.path.realpath(__file__))[0] + '/'


def get_all_readers(reader_list, exp_info):
    """根据 reader.yaml 注册表实例化本地 reader（单参构造）并调用指定方法。"""
    reader_register_dict = yaml.load(open(PWD + 'reader.yaml'), Loader=yaml.FullLoader)
    reader_instance_dict = {}
    short_name_dict = {}

    for file_name, reader_class_name in reader_register_dict.items():
        short_name = file_name.replace('_reader', '')
        module = __import__(file_name)
        class_ = getattr(module, reader_class_name)
        instance = class_(exp_info)
        reader_instance_dict[short_name] = instance
        for fun in [item[0] for item in inspect.getmembers(class_, inspect.isfunction)]:
            short_name_dict.setdefault(fun, []).append(short_name)

    reader_df_dict = {}
    df_name_list = []
    for reader_name in reader_list:
        if '.' in reader_name:
            short_name, fun_name = reader_name.split('.')
        else:
            fun_name = reader_name
            if fun_name not in short_name_dict or len(short_name_dict[fun_name]) > 1:
                print('function =', fun_name, short_name_dict, 'has conflict')
                return {}, []
            short_name = short_name_dict[fun_name][0]
        df, df_name = getattr(reader_instance_dict[short_name], fun_name)()
        reader_df_dict[df_name] = df
        df_name_list.append(df_name)
    return reader_df_dict, df_name_list
