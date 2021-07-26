import os
import yaml
import re


def read_param_yaml(model_name, window = False):
    if window:
        path = 'conf/params.yaml'
    else:
        path = r"/home/wyz/exp/NETL/conf/params.yaml"
    with open(path, 'r', encoding='utf-8') as f:
        dict = yaml.load(f.read(), Loader=yaml.FullLoader)[model_name]
    params_name = [key for key in dict.keys()]
    return params_name, dict

def get_done_combination(source_city, target_city, model_name, num_param, train_mode, window = False):
    if window:
        path = "E:\\code\\netl\\inter_data\\result\\{}_{}_{}_{}".format(model_name, train_mode, source_city, target_city)
    else:
        path = r"/home/wyz/exp/NETL/inter_data/result/{}_{}_{}_{}".format(model_name, train_mode, source_city, target_city)
    if not os.path.exists(path):
        return set()

    pattern_faction = re.compile(r'\d\.\d+')
    pattern_bool = re.compile('(True|False)')
    pattern_science = re.compile(r'\de-0\d')
    combinations = set()

    with open(path, 'r') as f:
        for i, content in enumerate(f.readlines()):
            if i == 0:
                continue
            content = '_'.join(re.split('\s+', content.strip())[:num_param])
            # fraction --> xe-x
            result_faction = re.search(pattern_faction, content)
            if result_faction is not None:
                science_model = re.sub(r'\.0+', '', "%e" % float(result_faction.group()))
                science_model = re.sub(r'-\d', '-', science_model)
                content = re.sub(pattern_faction, science_model, content)

            result_science = re.search(pattern_science, content)
            if result_science is not None:
                content = re.sub(r'-0', '-', content)

            # True | False --> 1 | 0
            result_bool = re.search(pattern_bool, content)
            if result_bool is not None:
                if result_bool.group() == 'True':
                    content = re.sub(pattern_bool, "1", content)
                else:
                    content = re.sub(pattern_bool, "0", content)

            combinations.add(content)
    return combinations

def get_undone_combination(dict, params_name, done_combination):
    def dfs(param_path, res):
        if len(param_path) == len(params_name):
            command = '_'.join(param_path)
            if command not in done_combination:
                command_dict = {k:v for k, v in zip(params_name, param_path)}
                res.append(command_dict)
            return
        cur_idx = len(param_path)
        for value in dict[params_name[cur_idx]]:
            param_path.append(str(value))
            dfs(param_path, res)
            del param_path[cur_idx]
    res = []
    dfs([], res)
    return res

def get_param_combination(source_city, target_city, model_name, train_mode, window=False):
    params_name, dict = read_param_yaml(model_name, window)
    num_param = len(params_name)
    done_combination = get_done_combination(source_city, target_city, model_name, num_param, train_mode, window)
    undone_combination = get_undone_combination(dict, params_name, done_combination)
    return undone_combination

if __name__ == '__main__':
    res = get_param_combination('beijing', 'tianjing', 'test')
    print(res)
