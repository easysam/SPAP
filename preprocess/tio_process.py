import datetime
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

exp_data_path = "data/exp_data"
station_count = {"beijing": 137, "guangzhou": 123, "tianjing": 101}

SEMI_GENERAL_HEADER = ['slow_num', 'fast_num', 'total_num'] + \
    ['hospital_f', 'spot_f', 'government_f', 'airport_f', 'subway_f', 'bus_f', 'bank_f', 'enterprise_f', 'school_f', 'community_f', 'hotel_f', 'supermarket_f', 'fast_food_f'] + \
    ['hospital_n', 'spot_n', 'government_n', 'airport_n', 'subway_n', 'bus_n', 'bank_n', 'enterprise_n', 'school_n', 'community_n', 'hotel_n', 'supermarket_n', 'fast_food_n'] + \
    ['hospital_p', 'spot_p', 'government_p', 'airport_p', 'subway_p', 'bus_p', 'bank_p', 'enterprise_p', 'school_p', 'community_p', 'hotel_p', 'supermarket_p', 'fast_food_p'] + \
    ['street_length', 'intersection_density_km', 'street_density_km', 'degree_centrality_avg'] + \
    ['subway', 'bus', 'park1', 'park2', 'park3', 'park4'] + \
    ['supermarket', 'mall']

FULL_GENERAL_HEADER = ['near_station', 'near_stub', 'slow_num', 'fast_num', 'total_num'] + \
    ['hospital_f', 'spot_f', 'government_f', 'airport_f', 'subway_f', 'bus_f', 'bank_f', 'enterprise_f', 'school_f', 'community_f', 'hotel_f', 'supermarket_f', 'fast_food_f'] + \
    ['hospital_n', 'spot_n', 'government_n', 'airport_n', 'subway_n', 'bus_n', 'bank_n', 'enterprise_n', 'school_n', 'community_n', 'hotel_n', 'supermarket_n', 'fast_food_n'] + \
    ['hospital_p', 'spot_p', 'government_p', 'airport_p', 'subway_p', 'bus_p', 'bank_p', 'enterprise_p', 'school_p', 'community_p', 'hotel_p', 'supermarket_p', 'fast_food_p'] + \
    ['street_length', 'intersection_density_km', 'street_density_km', 'degree_centrality_avg'] + \
    ['subway', 'bus', 'park1', 'park2', 'park3', 'park4'] + \
    ['supermarket', 'mall']


def static_feature(city, profiles=None):
    poi_frequency = np.load(exp_data_path + os.sep + 'poi_frequency' + os.sep + 'poi_frequency_{}.npy'.format(city),
                            allow_pickle=True)  # .tolist()
    poi_num = np.load(exp_data_path + os.sep + 'poi' + os.sep + 'poi_{}.npy'.format(city), allow_pickle=True)
    poi_entropy = np.load(exp_data_path + os.sep + 'poi_entropy' + os.sep + 'poi_entropy_{}.npy'.format(city),
                          allow_pickle=True)
    road = np.load(exp_data_path + os.sep + 'roadnet' + os.sep + 'roadnet_{}.npy'.format(city), allow_pickle=True)
    trans = np.load(exp_data_path + os.sep + 'transportation' + os.sep + 'transportation_{}.npy'.format(city),
                             allow_pickle=True)
    commerce = np.load(exp_data_path + os.sep + 'commerce' + os.sep + 'commerce_{}.npy'.format(city), allow_pickle=True)
    if profiles is None:
        demand_data = np.load("data/exp_data/station/all_demand_{}.npy".format(city), allow_pickle=True)
        ova_num = demand_data[:, 0, -2, np.newaxis]
        alt_num = demand_data[:, 0, 0, np.newaxis]
        dir_num = demand_data[:, 0, 2, np.newaxis]
    else:
        ova_num = profiles[:, 0, np.newaxis] + profiles[:, 1, np.newaxis]
        alt_num = profiles[:, 0, np.newaxis]
        dir_num = profiles[:, 1, np.newaxis]
    
    raw_data = np.concatenate((alt_num, dir_num, ova_num, poi_frequency, poi_num, poi_entropy, road, trans, commerce), axis=1)
    csv_data = pd.DataFrame(raw_data, columns=SEMI_GENERAL_HEADER)
    return csv_data, np.concatenate((alt_num, dir_num, ova_num), axis=1)


def tio_data_set(logger, city, static_feature, profiles, distances):
    # the overall_data is "static/full_neighbor_demand_{}.csv"
    profiles = pd.DataFrame(profiles, columns=["alt_num", "dir_num", "ova_num"])
    
    # get station set with more than 10 stub
    profiles = profiles.loc[profiles["ova_num"] > 0]
    # station_set is a mapper, map shift -> original_idx
    station_set = profiles.index

    station_cnt = len(station_set)

    """profile index is REINDEX after filter small station"""
    profiles.reset_index(drop=True, inplace=True)

    # calculate distances to all other stations for each station,
    # then select 10 nearest stations('s index)
    
    # slice station in station_set
    logger.info("# Making demand object data set.")
    valid_station_idx = np.array(station_set)
    logger.info("valid station len(stub num > {}): {}.".format(5, len(valid_station_idx)))
    logger.info("Shrinking distance matrix.")
    valid_station_distances = distances[valid_station_idx[:, np.newaxis], valid_station_idx]
    station_order = np.argsort(valid_station_distances[:, ::-1])
    # statistically count near station number and near stub number
    near_station_cnt = (valid_station_distances > 1/3).sum(axis=1)
    near_stub_cnt = np.zeros(station_cnt)
    for i in range(station_cnt):
        if near_station_cnt[i] > 0:
            near_stub_cnt[i] = profiles.loc[valid_station_distances[i] > 1/3, "ova_num"].sum()
    
	# Below is wrote by Yajie Ren, same as preprocess/process.py
	# 11 times header, get static neighborhood feature for each station(in S), get csv: full_neighbor_feature_{city}.csv
    ALL_HEADER = ['index']
    ALL_HEADER.extend(FULL_GENERAL_HEADER)
    for i in range(10):
        for j in FULL_GENERAL_HEADER:
            ALL_HEADER.append('{}_{}'.format(j, i))

    raw_data = np.empty((len(valid_station_distances), len(ALL_HEADER)))
    for i in range(station_cnt):
        raw_data[i][0] = station_set[i]
        raw_data[i][1] = near_station_cnt[i]
        raw_data[i][2] = near_stub_cnt[i]
        raw_data[i][3:3 + len(SEMI_GENERAL_HEADER)] = static_feature.iloc[i]['slow_num':'mall']
        for j in range(3):
            idx = station_order[i, j]
            raw_data[i][1 + len(FULL_GENERAL_HEADER) * (j + 1)] = near_station_cnt[idx]
            raw_data[i][1 + len(FULL_GENERAL_HEADER) * (j + 1) + 1] = near_stub_cnt[idx]
            raw_data[i][1 + len(FULL_GENERAL_HEADER) * (j + 1) + 2: 1 + len(FULL_GENERAL_HEADER) * (j + 2)] = \
                    static_feature.iloc[idx]['slow_num':'mall']
    neighbor_feature_data = pd.DataFrame(raw_data, columns=ALL_HEADER)
    logger.info("Neighbor static feature shape: {}".format(neighbor_feature_data.shape))

    # create final csv(11 times header with basic info(time_index + time_embed_index))
    # if index in S, fill basic info, neighbor_feature and demand

    demand = np.load(exp_data_path + os.sep + 'station' + os.sep + 'demand_{}.npy'.format(city), allow_pickle=True)
    time_count = demand.shape[1]

    DEMAND_HEADER = []
    DEMAND_HEADER.extend(ALL_HEADER)
    DEMAND_HEADER.extend(['time_index', 'time_embed', 'slow_demand', 'fast_demand', 'total_demand'])
    logger.info("Data set shape: valid station len * time cnt, header len: {} * {}, {}.".format(
        len(valid_station_distances), time_count, len(DEMAND_HEADER)
        ))
    neighbor_demand_raw_data = np.empty(((len(valid_station_distances) * time_count, len(DEMAND_HEADER))))

    # get time map like {"0800": 1, "0830": 2, ....}
    time_index_map = np.load(exp_data_path + os.sep + 'station_list' + os.sep + 'time_index.npy', allow_pickle=True)
    time_index_map = dict(time_index_map.tolist())
    time_map = {t: i for i, t in enumerate(sorted(set([k[-4:] for k in time_index_map['rev_index'].keys()])))}

    logger.info("Building demand object tensor")
    cur_idx = 0
    for time_idx in tqdm(range(time_count)):
        time_embed_idx = time_map[time_index_map['index'][time_idx][-4:]]
        for station_idx in station_set:
            neighbor_demand_raw_data[cur_idx][0:len(ALL_HEADER)] = neighbor_feature_data.loc[
                                                                   neighbor_feature_data['index'] == station_idx,
                                                                   'index':'mall_9']
            neighbor_demand_raw_data[cur_idx][len(ALL_HEADER)] = time_idx
            neighbor_demand_raw_data[cur_idx][len(ALL_HEADER) + 1] = time_embed_idx
            neighbor_demand_raw_data[cur_idx][len(ALL_HEADER) + 2] = demand[station_idx][time_idx][0]
            neighbor_demand_raw_data[cur_idx][len(ALL_HEADER) + 3] = demand[station_idx][time_idx][1]
            neighbor_demand_raw_data[cur_idx][len(ALL_HEADER) + 4] = demand[station_idx][time_idx][2]
            # todo add slow demand and quick demand here
            cur_idx = cur_idx + 1
    logger.info("Done! Shape is {}".format(neighbor_demand_raw_data.shape))
    neighbor_demand_data = pd.DataFrame(neighbor_demand_raw_data, columns=DEMAND_HEADER)
    # print(neighbor_demand_data.iloc[:3])

    return neighbor_demand_data


def inference_data_set(city, static_feature, profiles, distances):
    profiles = pd.DataFrame(profiles, columns=["alt_num", "dir_num", "ova_num"])
    station_cnt = len(static_feature.index)
    station_order = np.argsort(distances[:, ::-1])
    
    # statistically count near station number and near stub number
    near_station_cnt = (distances > 1/3).sum(axis=1)
    near_stub_cnt = np.zeros(station_cnt)
    for i in range(station_cnt):
        if near_station_cnt[i] > 0:
            near_stub_cnt[i] = profiles.loc[distances[i] > 1/3, "ova_num"].sum()
    
    # Below is wrote by Yajie Ren, same as preprocess/process.py
    # 11 times header, get static neighborhood feature for each station(in S), get csv: full_neighbor_feature_{city}.csv
    ALL_HEADER = ['index']
    ALL_HEADER.extend(FULL_GENERAL_HEADER)
    for i in range(10):
        for j in FULL_GENERAL_HEADER:
            ALL_HEADER.append('{}_{}'.format(j, i))

    raw_data = np.empty((len(static_feature.index), len(ALL_HEADER)))
    for i in range(station_cnt):
        raw_data[i][0] = i
        raw_data[i][1] = near_station_cnt[i]
        raw_data[i][2] = near_stub_cnt[i]
        raw_data[i][3:3 + len(SEMI_GENERAL_HEADER)] = static_feature.iloc[i]['slow_num':'mall']
        for j in range(10):
            idx = station_order[i, j]
            raw_data[i][1 + len(FULL_GENERAL_HEADER) * (j + 1)] = near_station_cnt[idx]
            raw_data[i][1 + len(FULL_GENERAL_HEADER) * (j + 1) + 1] = near_stub_cnt[idx]
            raw_data[i][1 + len(FULL_GENERAL_HEADER) * (j + 1) + 2: 1 + len(FULL_GENERAL_HEADER) * (j + 2)] = \
                    static_feature.iloc[idx]['slow_num':'mall']
    neighbor_feature_data = pd.DataFrame(raw_data, columns=ALL_HEADER)

    DEMAND_HEADER = []
    DEMAND_HEADER.extend(ALL_HEADER)
    DEMAND_HEADER.extend(['time_index', 'time_embed', 'slow_demand', 'fast_demand', 'total_demand'])


    time_key = []
    begin_time = datetime.datetime(2020, 1, 1, 8, 0)
    end_time = datetime.datetime(2020, 1, 1, 20, 30)
    while begin_time <= end_time:
        time_key.append(begin_time.strftime("%H%M"))
        begin_time += datetime.timedelta(minutes=30)
    
    # raw_object is raw feature with demand label
    raw_object = np.empty(((len(static_feature.index) * len(time_key), len(DEMAND_HEADER))))
    row_num = 0
    for idx, key in enumerate(time_key):
        for station_idx in range(station_cnt):
            raw_object[row_num][0: len(ALL_HEADER)] = neighbor_feature_data.loc[
                                                        neighbor_feature_data["index"] == station_idx,
                                                        "index": "mall_9"]
            raw_object[row_num][len(ALL_HEADER)] = idx
            raw_object[row_num][len(ALL_HEADER) + 1] = idx
            raw_object[row_num][len(ALL_HEADER) + 2] = 0
            raw_object[row_num][len(ALL_HEADER) + 3] = 0
            raw_object[row_num][len(ALL_HEADER) + 3] = 0
            row_num += 1

    neighbor_demand_data = pd.DataFrame(raw_object, columns=DEMAND_HEADER)

    return neighbor_demand_data

