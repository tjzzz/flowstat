#encoding=utf8
"""通用示例 metric：对 events DataFrame 按楼层过滤并计数（演示插件接口）。"""


class MetricDemo(object):
    def process(self, input, par_dict=None):
        par_dict = par_dict or {}
        events = input['events']
        floor = par_dict.get('floor', '')
        if floor:
            events = events[events['floor'] == floor]
        result = len(events)
        print('事件数=', result)
        return {
            'events': events,
            'result_num': result,
        }
