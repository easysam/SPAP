import sys
import os
import numpy as np
import copy
sys.path.append('..')

import yaml

# data_path = '/home/wyz/exp/NETL/data'
data_path = '/data'
exp_data_path = data_path + os.sep + 'exp_data'
station_count = {'beijing': 137, 'guangzhou': 123, 'tianjing': 101}

def read_yaml(windows = False):
    if windows:
        path = 'conf/file.yaml'
    else:
        path = 'conf/file.yaml'
    with open(path, 'r', encoding='utf-8') as f:
        files = yaml.load(f.read(), Loader=yaml.FullLoader)
    return files

def load_demand_and_hour(city):
    """
    load demand data
    :param city:
    :return:
    """
    time_index_map = np.load(exp_data_path + os.sep + 'station_list' + os.sep + 'time_index.npy', allow_pickle=True)
    time_index_map = dict(time_index_map.tolist())
    time_len = len(set([k[-4:] for k in time_index_map['rev_index'].keys()]))

    # load demand average data
    file_name = exp_data_path + os.sep + 'station' + os.sep + 'demand_avg_{}.npy'.format(city)
    demand_data = np.load(file_name, allow_pickle=True)

    hour, y = [], []
    for i in range(0, time_len):
        hour.append(i)
        y.append(demand_data[:, i, :])
    return hour, y

def load_matrix(city):
    matrix_distance = np.load(exp_data_path + os.sep + 'similarity' + os.sep + 'similarity_distance_{}_numpy.npy'.format(city), allow_pickle=True)
    matrix_poi = np.load(exp_data_path + os.sep + 'similarity' + os.sep + 'similarity_poi_frequency_{}_numpy.npy'.format(city), allow_pickle=True)
    matrix_roadnet = np.load(exp_data_path + os.sep + 'similarity' + os.sep + 'similarity_roadnet_{}_numpy.npy'.format(city), allow_pickle=True)
    return matrix_poi, matrix_distance, matrix_roadnet

def load_poi_related(city):
    poi_frequency = np.load(exp_data_path + os.sep + 'poi_frequency' + os.sep + 'poi_frequency_{}.npy'.format(city), allow_pickle=True)#.tolist()
    poi_num = np.load(exp_data_path + os.sep + 'poi' + os.sep + 'poi_{}.npy'.format(city), allow_pickle=True)
    poi_entropy = np.load(exp_data_path + os.sep + 'poi_entropy' + os.sep + 'poi_entropy_{}.npy'.format(city), allow_pickle=True)
    road = np.load(exp_data_path + os.sep + 'roadnet' + os.sep + 'roadnet_{}.npy'.format(city), allow_pickle=True)
    transportation = np.load(exp_data_path + os.sep + 'transportation' + os.sep + 'transportation_{}.npy'.format(city), allow_pickle=True)
    commerce = np.load(exp_data_path + os.sep + 'commerce' + os.sep + 'commerce_{}.npy'.format(city), allow_pickle=True)
    # print(road)
    # make road data into a similar magnitude
    road[:, 0] = road[:, 0] * 10
    road[:, 1] = road[:, 1] * 100
    road[:, 3] = road[:, 3] * 10000
    # print(road)
    """
    map = {}
    map['poi_frequency'] = [np.min(poi_frequency), np.max(poi_frequency)]
    map['poi_num'] = [np.min(poi_num), np.max(poi_num)]
    map['poi_entropy'] = [np.min(poi_entropy), np.max(poi_entropy)]
    map['road'] = [np.min(road), np.max(road)]
    map['transportation'] = [np.min(transportation), np.max(transportation)]
    map['commerce'] = [np.min(commerce), np.max(commerce)]
    avg_data_file_name = exp_data_path + os.sep + 'station' + os.sep + 'min_max'
    if os.path.exists(avg_data_file_name):
        os.remove(avg_data_file_name)
    np.save(avg_data_file_name, map)
    print(road)
    """

    min_max_map = np.load(exp_data_path + os.sep + 'poi' + os.sep + 'min_max.npy', allow_pickle=True)
    min_max_map = dict(min_max_map.tolist())
    # for k, v in min_max_map.items():
    #    print(k, v)
    poi_frequency = (poi_frequency - min_max_map['poi_frequency'][0]) / (min_max_map['poi_frequency'][1] - min_max_map['poi_frequency'][0])
    poi_num = (poi_num - min_max_map['poi_num'][0]) / (min_max_map['poi_num'][1] - min_max_map['poi_num'][0])
    poi_entropy = (poi_entropy - min_max_map['poi_entropy'][0]) / (min_max_map['poi_entropy'][1] - min_max_map['poi_entropy'][0])
    road = (road - min_max_map['road'][0]) / (min_max_map['road'][1] - min_max_map['road'][0])
    transportation = (transportation - min_max_map['transportation'][0]) / (min_max_map['transportation'][1] - min_max_map['transportation'][0])
    commerce = (commerce - min_max_map['commerce'][0]) / (min_max_map['commerce'][1] - min_max_map['commerce'][0])
    return poi_frequency, poi_num, poi_entropy, road, transportation, commerce


def load_demand_and_feature(city):
    hour, y = load_demand_and_hour(city)
    matrix_poi, matrix_distance, matrix_roadnet = load_matrix(city)
    poi_frequency, poi_num, poi_entropy, road, transportation, commerce = load_poi_related(city)

    static_map = {'poi_frequency': poi_frequency, 'poi_num': poi_num, 'poi_entropy': poi_entropy,
                  'road': road, 'transportation': transportation, 'commerce': commerce}
    x = []
    for h in hour:
        map = copy.deepcopy(static_map)
        map['hour'] = h
        x.append(map)
    return x, y, [matrix_poi, matrix_distance, matrix_roadnet]



if __name__ == '__main__':
    load_demand_and_hour('beijing')
    pass
