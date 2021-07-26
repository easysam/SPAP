import numpy as np
import csv
import shutil
import os
import re
import datetime
import math
from geopy.distance import geodesic
import pandas as pd
import pickle

data_path = '/home/ryj/renyajie/exp/NETL/data'
# data_path = r'E:\code\netl\data'
spider_data_path = data_path + os.sep + 'spider_data'
exp_data_path = data_path + os.sep + 'exp_data'
data_length = {'beijing': 138, 'guangzhou': 123, 'tianjing': 101}

def check_path(str):
    if not os.path.exists(str):
        os.mkdir(str)

def filter_demand(city, ratio):
    """
    filter charge demand data less than given ratio
    :param city: which city
    :param ratio: fraction, like 0.9 or 0.8
    :return:
    """
    check_path(exp_data_path + os.sep + 'station')
    check_path(exp_data_path + os.sep + 'station' + os.sep + city)
    for file in os.listdir(exp_data_path + os.sep + 'station' + os.sep + city):
        os.remove(exp_data_path + os.sep + 'station' + os.sep + city + os.sep + file)

    for file in os.listdir(spider_data_path + os.sep + 'station' + os.sep + city):
        count = 0
        less_ten_count = 0
        with open(spider_data_path + os.sep + 'station' + os.sep + city + os.sep + file, "r") as f:
            reader = csv.reader(f)
            for line in reader:
                count = count + 1
                if len(line) < 10:
                    less_ten_count = less_ten_count + 1
        if count / data_length[city] >= ratio and less_ten_count / data_length[city] <= 0.1:
            shutil.copy(spider_data_path + os.sep + 'station' + os.sep + city + os.sep + file,
                        exp_data_path + os.sep + 'station' + os.sep + city + os.sep + file)

    print('before', len(os.listdir(spider_data_path + os.sep + 'station' + os.sep + city)))
    print('after', len(os.listdir(exp_data_path + os.sep + 'station' + os.sep + city)))

def get_top_station_set(city):
    """
    get most frequent station, return a new map(relation from old_index to new index(0-indexed))
    :param city: which city
    :return: a new relation map
    """
    s = {}
    for file in os.listdir(exp_data_path + os.sep + 'station' + os.sep + city):
        with open(exp_data_path + os.sep + 'station' + os.sep + city + os.sep + file) as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] not in s:
                    s[row[0]] = 1
                else:
                    s[row[0]] = s[row[0]] + 1

    sort_s = dict(sorted(s.items(), key=lambda x : x[1], reverse=True))
    first = True
    res = []
    for k, v in sort_s.items():
        if first:
            top = v
            first = False
        if top - v <= 30:
            res.append(k)
    print('before', len(sort_s))
    print('after', len(res))

    # restore new map [old_index, new_index]
    list_remap = {}
    new_index = 0
    for index in range(0, data_length[city]):
        if str(index) in res:
            list_remap[index] = new_index
            new_index = new_index + 1

    # print(list_remap)
    check_path(exp_data_path + os.sep + 'station_list')
    file_name = exp_data_path + os.sep + 'station_list' + os.sep + 'list_remap_{}'.format(city) + '.npy'
    if os.path.exists(file_name):
        os.remove(file_name)
    np.save(file_name, list_remap)

def get_top_station_csv(city):
    """
    get station set for every citys
    :param city:
    :return:
    """
    list_index = np.load("/home/ryj/renyajie/exp/NETL/data/spider_data/station_list/final_list_index.npy",
                         allow_pickle=True)
    list_remap = np.load("/home/ryj/renyajie/exp/NETL/data/exp_data/station_list/list_remap_{}.npy".format(city),
                         allow_pickle=True)
    list_index = dict(list_index.tolist())
    list_remap = dict(list_remap.tolist())

    # get longitude and latitude
    geo_map = {}
    with open("/home/ryj/renyajie/exp/NETL/data/spider_data/station_list/final_list_{}.csv".format(city)) as f:
        reader = csv.reader(f)
        for line in reader:
            geo_map[int(line[0])] = (float(line[1]), float(line[2]))

    with open("/home/ryj/renyajie/exp/NETL/data/exp_data/station_list/list_{}.csv".format(city), "w") as f:
        writer = csv.writer(f)
        for old_index, new_index in list_remap.items():
            if old_index in list_index[city]:
                writer.writerow([new_index, geo_map[old_index][0], geo_map[old_index][1], list_index[city][old_index]])
    pass

def get_time_index():
    """
    get time index for every city
    :return:
    """

    def get_common_list(map):
        res = set()
        first = True
        for _, value in map.items():
            if first:
                res = set(value)
                first = False
            else:
                res = res.intersection(value)
        return list(res)

    # get common time for 3 cities
    demand_map = {}
    for city in ['beijing', 'tianjing', 'guangzhou']:
        demand_map[city] = []
        for file in os.listdir(exp_data_path + os.sep + 'station' + os.sep + city):
            demand_map[city].append(file[0:12])


    # get weather time
    weather_map = {}
    for city in ['beijing', 'tianjing', 'guangzhou']:
        weather_map[city] = []
        for file in os.listdir(spider_data_path + os.sep + 'weather' + os.sep + city):
            with open(spider_data_path + os.sep + 'weather' + os.sep + city + os.sep + file) as f:
                reader = csv.reader(f)
                for line in reader:
                    # t = line[0].replace("\ufeff", "")
                    weather_map[city].append(line[0].replace("\ufeff", ""))

    demand_res = get_common_list(demand_map)
    weather_res = get_common_list(weather_map)
    res = get_common_list({'demand': demand_res, 'weather': weather_res})
    res = sorted(res)
    print(len(res))
    print(res)


    time_index = {'index': {}, 'rev_index': {}}
    index = 0
    for time in res:
        time_index['index'][index] = time
        time_index['rev_index'][time] = index
        index = index + 1

    file_name = exp_data_path + os.sep + 'station_list' + os.sep + 'time_index.npy'
    if os.path.exists(file_name):
        os.remove(file_name)
    np.save(file_name, time_index)

def get_city_weather_and_dispatch(city):
    """
    accumulate city weather into a file, then dispatch into concrete matter with time-index(0-indexed)
    :param city:
    :return: a whole file with all records and some detailed csv file
    """
    # accumulate weather record for city
    check_path(exp_data_path + os.sep + 'weather')
    check_path(exp_data_path + os.sep + 'weather' + os.sep + city)
    time_index = np.load(exp_data_path + os.sep + 'station_list' + os.sep + 'time_index.npy', allow_pickle=True)
    time_index = dict(time_index.tolist())
    with open(exp_data_path + os.sep + 'weather' + os.sep + city + os.sep + 'whole_{}.csv'.format(city), "w") as f:
        writer = csv.writer(f)
        res = []
        for file in os.listdir(spider_data_path + os.sep + 'weather' + os.sep + city):
            with open(spider_data_path + os.sep + 'weather' + os.sep + city + os.sep + file) as day:
                reader = csv.reader(day)
                for line in reader:
                    line[0] = line[0].replace("\ufeff", "")
                    if line[0] in time_index['rev_index']:
                        line.insert(0, time_index['rev_index'][line[0]])
                        res.append(line)
        res = sorted(res, key=lambda x : x[0])
        writer.writerows(res)

    # replace index and divide into concrete matter
    name_pos = {'weather_type': 2, 'temperature': 3, 'air': 4, 'wind': 6}
    con_res = {}
    with open(exp_data_path + os.sep + 'weather' + os.sep + city + os.sep + 'whole_{}.csv'.format(city)) as f:
        reader = csv.reader(f)
        for line in reader:
            for name, index in name_pos.items():
                if name not in con_res:
                    con_res[name] = []
                con_res[name].append([line[0], line[index]])

    for name, index in name_pos.items():
        with open(exp_data_path + os.sep + 'weather' + os.sep + city + os.sep + '{}_{}.csv'.format(city, name), "w") as f:
            writer = csv.writer(f)
            for line in con_res[name]:
                writer.writerow(line)

def check_all_type(name):
    """
    check how many type for certain aspect
    :return:
    """
    all_type = set()
    for city in ['beijing', 'tianjing', 'guangzhou']:
        with open(
                exp_data_path + os.sep + 'weather' + os.sep + city + os.sep + '{}_{}.csv'.format(city, name)) as f:
            reader = csv.reader(f)
            for line in reader:
                all_type.add(line[1].replace(" ", ""))
    print(all_type)

def weather_of_weather_type(city):
    """
    separate weather type, which use number(start from 0) to represent those main weather types
    :param city:
    :return: a weather npy type file
    """
    # get all weather type, and make a map relation
    check_all_type('weather_type')
    relation_type_map = {'阴': '阴', '小雨转阴': '小雨', '中雨': '中雨', '小雨': '小雨',
                    '多云转小雨': '小雨', '中雪': '雪', '小雨转雪': '小雨',
                    '多云': '多云', '霾': '霾', '晴': '晴', '阴转小雨': '小雨'}
    relation_index = {big_type: index for index, big_type in enumerate(sorted(set(relation_type_map.values())))}
    relation_index_map = {small_type: relation_index[big_type] for small_type, big_type in relation_type_map.items()}
    # use relation map to get a weather npy file

    time_index = np.load(exp_data_path + os.sep + 'station_list' + os.sep + 'time_index.npy', allow_pickle=True)
    time_index = dict(time_index.tolist())
    numpy_res = np.empty((len(time_index['index']),))
    with open(exp_data_path + os.sep + 'weather' + os.sep + city + os.sep + '{}_weather_type.csv'.format(city)) as f:
        reader = csv.reader(f)
        for line in reader:
            numpy_res[int(line[0])] = relation_index_map[line[1]]

    file_name = exp_data_path + os.sep + 'weather' + os.sep + city + os.sep + '{}_weather_type'.format(city)
    if os.path.exists(file_name):
        os.remove(file_name)
    np.save(file_name, numpy_res)

def weather_of_temperature(city):
    """
    separate temperature
    :param city:
    :return: a temperature npy type file
    """
    pattern = re.compile(r'(-?\d+).*')

    time_index = np.load(exp_data_path + os.sep + 'station_list' + os.sep + 'time_index.npy', allow_pickle=True)
    time_index = dict(time_index.tolist())
    numpy_res = np.empty((len(time_index['index']),))
    with open(exp_data_path + os.sep + 'weather' + os.sep + city + os.sep + '{}_temperature.csv'.format(city)) as f:
        reader = csv.reader(f)
        for line in reader:
            line[1] = pattern.match(line[1]).group(1)
            numpy_res[int(line[0])] = int(line[1])

    file_name = exp_data_path + os.sep + 'weather' + os.sep + city + os.sep + '{}_temperature'.format(city)
    if os.path.exists(file_name):
        os.remove(file_name)
    np.save(file_name, numpy_res)
    pass

def weather_of_wind(city):
    """
    separate wind speed, which use number(start from 0) to represent those speed
    :param city:
    :return: a wind file
    """
    pattern = re.compile(r'.*(\d+).*')

    time_index = np.load(exp_data_path + os.sep + 'station_list' + os.sep + 'time_index.npy', allow_pickle=True)
    time_index = dict(time_index.tolist())
    numpy_res = np.empty((len(time_index['index']),))
    with open(exp_data_path + os.sep + 'weather' + os.sep + city + os.sep + '{}_wind.csv'.format(city)) as f:
        reader = csv.reader(f)
        for line in reader:
            if '微' in line[1]:
                line[1] = 0
            else:
                line[1] = pattern.match(line[1]).group(1)
            numpy_res[int(line[0])] = int(line[1])

    file_name = exp_data_path + os.sep + 'weather' + os.sep + city + os.sep + '{}_wind'.format(city)
    if os.path.exists(file_name):
        os.remove(file_name)
    np.save(file_name, numpy_res)
    pass

def weather_of_air(city):
    """
    separate air quality, which use number(start from 0) to represent those quality
    :param city:
    :return: a air quality file
    """

    # check_all_type('air')
    relation_index_map = {'优': 0, '良好': 1,  '轻度污染': 2, '中度污染': 3, '重度污染': 4}

    time_index = np.load(exp_data_path + os.sep + 'station_list' + os.sep + 'time_index.npy', allow_pickle=True)
    time_index = dict(time_index.tolist())
    numpy_res = np.empty((len(time_index['index']),))
    with open(exp_data_path + os.sep + 'weather' + os.sep + city + os.sep + '{}_air.csv'.format(city)) as f:
        reader = csv.reader(f)
        for line in reader:
            line[1] = line[1].replace(" ", "")
            numpy_res[int(line[0])] = relation_index_map[line[1]]

    file_name = exp_data_path + os.sep + 'weather' + os.sep + city + os.sep + '{}_air'.format(city)
    if os.path.exists(file_name):
        os.remove(file_name)
    np.save(file_name, numpy_res)
    pass

def weather_of_hour():
    """
    separate hour
    :param city:
    :return: a hour file
    """
    time_index = np.load(exp_data_path + os.sep + 'station_list' + os.sep + 'time_index.npy', allow_pickle=True)
    time_index = dict(time_index.tolist())
    numpy_res = np.empty((len(time_index['index']),))
    for index, time in time_index['index'].items():
        numpy_res[int(index)] = int(str(time)[8:10])

    file_name = exp_data_path + os.sep + 'weather' + os.sep + 'hour'
    if os.path.exists(file_name):
        os.remove(file_name)
    np.save(file_name, numpy_res)

def weather_of_day():
    """
    separate day(0-6)
    :param city:
    :return: a day file
    """
    time_index = np.load(exp_data_path + os.sep + 'station_list' + os.sep + 'time_index.npy', allow_pickle=True)
    time_index = dict(time_index.tolist())
    numpy_res = np.empty((len(time_index['index']),))
    for index, time in time_index['index'].items():
        year, month, day = str(time)[0:4], str(time)[4:6], str(time)[6:8]
        w = datetime.date(int(year), int(month), int(day))
        numpy_res[int(index)] = w.weekday()

    file_name = exp_data_path + os.sep + 'weather' + os.sep + 'day'
    if os.path.exists(file_name):
        os.remove(file_name)
    np.save(file_name, numpy_res)
    pass

def holiday():
    """
    get a mapping relation between time-index and holiday, which uses 1 to represent holiday(0-1)
    :return: a holiday file
    """
    time_index = np.load(exp_data_path + os.sep + 'station_list' + os.sep + 'time_index.npy', allow_pickle=True)
    time_index = dict(time_index.tolist())
    numpy_res = np.empty((len(time_index['index']),))
    for index, time in time_index['index'].items():
        if str(time)[0:8] == '20200101':
            numpy_res[int(index)] = 1
        else:
            numpy_res[int(index)] = 0

    file_name = exp_data_path + os.sep + 'weather' + os.sep + 'holiday'
    if os.path.exists(file_name):
        os.remove(file_name)
    np.save(file_name, numpy_res)
    pass

def filter_station(city):
    """
    remove unnecessary file
    :param city:
    :return:
    """
    time_index = np.load(exp_data_path + os.sep + 'station_list' + os.sep + 'time_index.npy', allow_pickle=True)
    time_index = dict(time_index.tolist())
    pattern = re.compile('(\d+).*')
    for file in os.listdir(exp_data_path + os.sep + 'station' + os.sep + city):
        name = pattern.match(file).group(1)
        if name not in time_index['rev_index']:
            os.remove(exp_data_path + os.sep + 'station' + os.sep + city + os.sep + file)

    # remove unnecessary record, and update station index
    list_remap = np.load("/home/ryj/renyajie/exp/NETL/data/exp_data/station_list/list_remap_{}.npy".format(city),
                         allow_pickle=True)
    list_remap = dict(list_remap.tolist())
    for file in os.listdir(exp_data_path + os.sep + 'station' + os.sep + city):
        res = []
        with open(exp_data_path + os.sep + 'station' + os.sep + city + os.sep + file) as of:
            reader = csv.reader(of)
            for line in reader:
                # remove missing data record and inactive station
                if len(line) < 10:
                    continue
                if int(line[0]) in list_remap:
                    line[0] = list_remap[int(line[0])]
                    res.append(line)
        with open(exp_data_path + os.sep + 'station' + os.sep + city + os.sep + "new_" + file, "w") as nf:
            writer = csv.writer(nf)
            for row in res:
                writer.writerow(row)

    for file in os.listdir(exp_data_path + os.sep + 'station' + os.sep + city):
        if not file.startswith("new_"):
            os.remove(exp_data_path + os.sep + 'station' + os.sep + city + os.sep + file)
    for file in os.listdir(exp_data_path + os.sep + 'station' + os.sep + city):
        prefix = exp_data_path + os.sep + 'station' + os.sep + city
        os.rename(prefix + os.sep + file, prefix + os.sep + file[4:])

station_count = {'beijing': 137, 'guangzhou': 123, 'tianjing': 101}
def make_demand_tensor(city):
    """
    change demand csv into npy file, under the constrain of station set and time index
    :param city:
    :return: a npy file
    """
    # remove file
    # filter_station(city)


    # calculate each time unit for each city, and get a average count map
    # 0. get station number map {station: [num1, num2, num3]}(num3 == num1 + num2)
    # 1. get all record into a map {time unit: {station: [amount1, count1, amount2, count2, amount3, count3]}}
    # 2. combine two maps, then change into {time unit: {station: [emp1, num1, r1, emp2, num2, r2, emp3, num3, r3]}}
    number_map = {}
    for file in os.listdir(exp_data_path + os.sep + 'station' + os.sep + city):
        with open(exp_data_path + os.sep + 'station' + os.sep + city + os.sep + file) as f:
            reader = csv.reader(f)
            for line in reader:
                if line[0] not in number_map:
                    number_map[line[0]] = [line[2],line[5],line[8]]
        if len(number_map) == station_count[city]:
            break

    count_map = {}
    for file in os.listdir(exp_data_path + os.sep + 'station' + os.sep + city):
        time_unit = file[8:12]
        if time_unit not in count_map:
            count_map[time_unit] = {}
        with open(exp_data_path + os.sep + 'station' + os.sep + city + os.sep + file) as f:
            reader = csv.reader(f)
            for line in reader:
                if len(line) < 10:
                    continue
                station_id = line[0]
                amount1, amount2, amount3 = float(line[3]), float(line[6]), float(line[9])
                if station_id not in count_map[time_unit]:
                    count_map[time_unit][station_id] = [0, 0, 0, 0, 0, 0]
                count_map[time_unit][station_id][0] += amount1
                count_map[time_unit][station_id][2] += amount2
                count_map[time_unit][station_id][4] += amount3
                count_map[time_unit][station_id][1] += 1
                count_map[time_unit][station_id][3] += 1
                count_map[time_unit][station_id][5] += 1

    average_map = {}
    for time_unit, station_map in count_map.items():
        average_map[time_unit] = {}
        for station_id, info in station_map.items():
            rate1 = count_map[time_unit][station_id][0] / count_map[time_unit][station_id][1]
            rate2 = count_map[time_unit][station_id][2] / count_map[time_unit][station_id][3]
            rate3 = count_map[time_unit][station_id][4] / count_map[time_unit][station_id][5]
            count1 = int(number_map[station_id][0])
            count2 = int(number_map[station_id][1])
            count3 = int(number_map[station_id][2])
            emp1, emp2, emp3 = math.ceil(rate1 * count1), math.ceil(rate2 * count2), math.ceil(rate3 * count3)
            average_map[time_unit][station_id] = [station_id, emp1, count1, rate1, emp2, count2, rate2, emp3, count3, rate3]


    # fill the missing record according to the above map
    num = 0
    for file in os.listdir(exp_data_path + os.sep + 'station' + os.sep + city):
        with open(exp_data_path + os.sep + 'station' + os.sep + city + os.sep + file) as f:
            count = len(f.readlines())
            if count == station_count[city]:
                continue
            num += 1
            print('fill for', file)
            print('before', count)

        with open(exp_data_path + os.sep + 'station' + os.sep + city + os.sep + file) as f:
            reader = csv.reader(f)
            a = [int(line[0]) for line in reader]

        with open(exp_data_path + os.sep + 'station' + os.sep + city + os.sep + file, "a") as f:
            writer = csv.writer(f)
            diff_set = set(range(0, station_count[city])) - set(a)
            time_unit = file[8:12]
            for station_id in sorted(diff_set):
                writer.writerow(average_map[time_unit][str(station_id)])

        with open(exp_data_path + os.sep + 'station' + os.sep + city + os.sep + file, "r") as f:
            print('after', len(f.readlines()))
    print('fill', num, 'files')


    # change csv file into npy file
    time_index_map = np.load(exp_data_path + os.sep + 'station_list' + os.sep + 'time_index.npy', allow_pickle=True)
    time_index_map = dict(time_index_map.tolist())

    numpy_all_data = np.empty((station_count[city], len(time_index_map['index']), 6))
    numpy_data = np.empty((station_count[city], len(time_index_map['index']), 3))
    for file in os.listdir(exp_data_path + os.sep + 'station' + os.sep + city):
        time_index = int(time_index_map['rev_index'][file[0:12]])
        with open(exp_data_path + os.sep + 'station' + os.sep + city + os.sep + file) as f:
            reader = csv.reader(f)
            for line in reader:
                station_id = int(line[0])
                try:
                    # 交流(慢充)，直流（慢充），总数
                    count1, count2, count3 = int(line[2]), int(line[5]), int(line[8])
                    # 慢充需求，快充需求，总需求
                    rate1, rate2, rate3 = float(line[3]), float(line[6]), float(line[9])
                except:
                    print(file, line)
                    break
                numpy_data[station_id, time_index] = [rate1, rate2, rate3]
                numpy_all_data[station_id, time_index] = [count1, rate1, count2, rate2, count3, rate3]

    data_file_name = exp_data_path + os.sep + 'station' + os.sep + 'demand_{}'.format(city)
    if os.path.exists(data_file_name):
        os.remove(data_file_name)
    np.save(data_file_name, numpy_data)
    print(numpy_data.shape)
    print(numpy_all_data.shape)

    all_data_file_name = exp_data_path + os.sep + 'station' + os.sep + 'all_demand_{}'.format(city)
    if os.path.exists(all_data_file_name):
        os.remove(all_data_file_name)
    np.save(all_data_file_name, numpy_all_data)
    pass

def filter_road_and_poi_and_save_npy(city, name, func):
    """
    filter record under the constrain of station set, and update station index
    :param city:
    :param func: float | int
    :return: a new csv file, which accords with station set, and npy file
    """
    check_path(exp_data_path + os.sep + name)
    list_remap = np.load("/home/ryj/renyajie/exp/NETL/data/exp_data/station_list/list_remap_{}.npy".format(city),
                         allow_pickle=True)
    list_remap = dict(list_remap.tolist())
    numpy_res = np.empty((station_count[city], property_length[name]))
    res = []
    with open(spider_data_path + os.sep + name + os.sep + '{}_{}.csv'.format(name, city)) as f:
        reader = csv.reader(f)
        for line in reader:
            if int(line[0]) in list_remap:
                line[0] = list_remap[int(line[0])]
                res.append(line)
                numpy_res[int(line[0])] = list(map(func, line[1:]))
    with open(exp_data_path + os.sep + name + os.sep + '{}_{}.csv'.format(name, city), "w") as f:
        writer = csv.writer(f)
        writer.writerows(res)
    print(len(res), numpy_res.shape)

    file_name = exp_data_path + os.sep + name + os.sep + '{}_{}'.format(name, city)
    if os.path.exists(file_name):
        os.remove(file_name)
    np.save(file_name, numpy_res)
    pass

def make_poi_frequency_and_entropy(city):
    """
    generate poi frequency and entropy in spider data according to poi
    :param city:
    :return: poi frequency and entropy csv
    """
    frequency_res, entropy_res = [], []
    with open(spider_data_path + os.sep + 'poi' + os.sep + 'poi_{}.csv'.format(city)) as f:
        reader = csv.reader(f)
        for line in reader:
            line = list(map(int, line))
            total = sum(line[1:])
            frequency_tmp = [line[0]]
            frequency_tmp.extend([num / total for num in line[1:]])
            entropy_tmp = [line[0]]
            entropy_tmp.extend([-freq * np.log2(freq) if freq > 0 else 0 for freq in frequency_tmp[1:]])
            frequency_res.append(frequency_tmp)
            entropy_res.append(entropy_tmp)

    with open(spider_data_path + os.sep + 'poi' + os.sep + 'poi_frequency_{}.csv'.format(city), "w") as f:
        writer = csv.writer(f)
        writer.writerows(frequency_res)
    with open(spider_data_path + os.sep + 'poi' + os.sep + 'poi_entropy_{}.csv'.format(city), "w") as f:
        writer = csv.writer(f)
        writer.writerows(entropy_res)
    pass

property_length = {'poi_frequency': 13, 'poi': 13, 'poi_entropy': 13, 'roadnet': 4, 'commerce': 2, 'transportation': 6}
def make_poi_frequency_and_road_similarity_matrix(city, name):
    """
    make poi and road similarity_matrix, by pearson
    :param city:
    :param name: poi frequency or road net
    :return: a numpy typed npy file
    """
    # make numpy array from npy file
    file = np.load("/home/ryj/renyajie/exp/NETL/data/exp_data/{}/{}_{}.npy".format(name, name, city),
                         allow_pickle=True)
    file = file.tolist()
    numpy_file = np.empty((station_count[city], property_length[name]))
    for line in file:
        numpy_file[int(line[0])] = list(map(float, line[1:]))
    similarity_matrix = np.corrcoef(numpy_file)*0.5 + 0.5

    check_path(exp_data_path + os.sep + 'similarity')
    file_name = exp_data_path + os.sep + 'similarity' + os.sep + 'similarity_{}_{}_numpy'.format(name, city)
    if os.path.exists(file_name):
        os.remove(file_name)
    np.save(file_name, similarity_matrix)
    pass

def make_distance_matrix(city):
    """
    make distance matrix according to latitude and longitude, 1 is divided by distance simply
    :param city:
    :return: a numpy typed npy file
    """
    distance_map = {}
    with open(spider_data_path + os.sep + 'station_list' + os.sep + 'list_{}.csv'.format(city)) as f:
        reader = csv.reader(f)
        for line in reader:
            distance_map[int(line[0])] = (float(line[1]), float(line[2]))

    numpy_file = np.empty((station_count[city], station_count[city]))
    for i in range(0, station_count[city]):
        for j in range(0, station_count[city]):
            if i == j:
                numpy_file[i, j] = 1
            else:
                distance = geodesic(distance_map[i], distance_map[j]).km
                numpy_file[i, j] = 1.0 / distance if distance > 1 else 1

    file_name = exp_data_path + os.sep + 'similarity' + os.sep + 'similarity_distance_{}_numpy'.format(city)
    if os.path.exists(file_name):
        os.remove(file_name)
    np.save(file_name, numpy_file)
    pass

def fix_demand(city):
    """
    make sure csv is ok, e3 = e1 + e2, o3 = o1 + o2
    :param city:
    :return:
    """
    for file in os.listdir(exp_data_path + os.sep + 'station' + os.sep + city):
        with open(exp_data_path + os.sep + 'station' + os.sep + city + os.sep + 'new_' + file, "w") as wf:
            writer = csv.writer(wf)
            with open(exp_data_path + os.sep + 'station' + os.sep + city + os.sep + file) as rf:
                reader = csv.reader(rf)
                for row in reader:
                    writer.writerow([int(row[0]), int(row[1]), int(row[2]), float(row[3]), int(row[4]), int(row[5]), float(row[6]),
                                     int(row[1]) + int(row[4]), int(row[2]) + int(row[5]), (int(row[2]) + int(row[5]) - int(row[1]) - int(row[4])) / (int(row[2]) + int(row[5]))])

    for file in os.listdir(exp_data_path + os.sep + 'station' + os.sep + city):
        if not file.startswith('new'):
            os.remove(exp_data_path + os.sep + 'station' + os.sep + city + os.sep + file)
    for file in os.listdir(exp_data_path + os.sep + 'station' + os.sep + city):
        prefix = exp_data_path + os.sep + 'station' + os.sep + city
        os.rename(prefix + os.sep + file, prefix + os.sep + file[4:])

def make_avg_demand(city):
    """
    change demand data into average data
    :param city:
    :return:
    """
    # get time map like {"0800": 1, "0830": 2, ....}
    time_index_map = np.load(exp_data_path + os.sep + 'station_list' + os.sep + 'time_index.npy', allow_pickle=True)
    time_index_map = dict(time_index_map.tolist())
    time_map = {t : i for i, t in enumerate(sorted(set([k[-4:] for k in time_index_map['rev_index'].keys()])))}

    # sum up all time for each station
    demand = np.load(exp_data_path + os.sep + 'station' + os.sep + 'demand_{}.npy'.format(city), allow_pickle=True)
    sum_demand = np.zeros((demand.shape[0], len(time_map), 4))
    for i in range(0, demand.shape[0]):
        for j in range(0, demand.shape[1]):
            for k in range(0, 3):
                sum_demand[i, time_map[time_index_map['index'][j][-4:]]][k] += demand[i,j,k]
            sum_demand[i, time_map[time_index_map['index'][j][-4:]]][3] += 1

    # get average demand
    avg_demand = np.zeros((demand.shape[0], len(time_map), 3))
    for i in range(0, demand.shape[0]):
        for j in range(0, len(time_map)):
            for k in range(0, 3):
                avg_demand[i,j,k] = sum_demand[i,j,k] / sum_demand[i,j,3]

    avg_data_file_name = exp_data_path + os.sep + 'station' + os.sep + 'demand_avg_{}'.format(city)
    if os.path.exists(avg_data_file_name):
        os.remove(avg_data_file_name)
    np.save(avg_data_file_name, avg_demand)

GENERAL_HEADER = ['num'] + \
                 ['hospital_f', 'spot_f', 'government_f', 'airport_f', 'subway_f', 'bus_f', 'bank_f', 'enterprise_f', 'school_f', 'community_f', 'hotel_f', 'supermarket_f', 'fast_food_f'] + \
                 ['hospital_n', 'spot_n', 'government_n', 'airport_n', 'subway_n', 'bus_n', 'bank_n', 'enterprise_n', 'school_n', 'community_n', 'hotel_n', 'supermarket_n', 'fast_food_n'] + \
                 ['hospital_p', 'spot_p', 'government_p', 'airport_p', 'subway_p', 'bus_p', 'bank_p', 'enterprise_p', 'school_p', 'community_p', 'hotel_p', 'supermarket_p', 'fast_food_p'] + \
                 ['street_length', 'intersection_density_km', 'street_density_km', 'degree_centrality_avg'] + \
                 ['subway', 'bus', 'park1', 'park2', 'park3', 'park4'] + \
                 ['supermarket', 'mall']
def get_all_station_feature(city):
    """
    make all static feature into a single file
    """
    poi_frequency = np.load(exp_data_path + os.sep + 'poi_frequency' + os.sep + 'poi_frequency_{}.npy'.format(city),
                            allow_pickle=True)  # .tolist()
    poi_num = np.load(exp_data_path + os.sep + 'poi' + os.sep + 'poi_{}.npy'.format(city), allow_pickle=True)
    poi_entropy = np.load(exp_data_path + os.sep + 'poi_entropy' + os.sep + 'poi_entropy_{}.npy'.format(city),
                          allow_pickle=True)
    road = np.load(exp_data_path + os.sep + 'roadnet' + os.sep + 'roadnet_{}.npy'.format(city), allow_pickle=True)
    transportation = np.load(exp_data_path + os.sep + 'transportation' + os.sep + 'transportation_{}.npy'.format(city),
                             allow_pickle=True)
    commerce = np.load(exp_data_path + os.sep + 'commerce' + os.sep + 'commerce_{}.npy'.format(city), allow_pickle=True)

    file_name = exp_data_path + os.sep + 'station' + os.sep + 'all_demand_{}.npy'.format(city)
    demand_data = np.load(file_name, allow_pickle=True)
    num = demand_data[:, 0, -2, np.newaxis] # todo check meaning here, get quick and slow feature

    raw_data = np.concatenate((num, poi_frequency, poi_num, poi_entropy, road, transportation, commerce), axis=1)
    csv_data = pd.DataFrame(raw_data, columns=GENERAL_HEADER)

    file_path = exp_data_path + os.sep + 'static' + os.sep + 'static_feature_{}.csv'.format(city)
    if os.path.exists(file_path):
        os.remove(file_path)
    csv_data.to_csv(file_path)
    pass

def get_neigh_demand(city):
    """
    get demand data from each station, each time. each record includes basic info, neighborhood feature and demand value
    """

    # get station set S with more than 10 charge equipment
    static_file_path = exp_data_path + os.sep + 'static' + os.sep + 'static_feature_{}.csv'.format(city)
    static_feature = pd.read_csv(static_file_path, header=0)
    station_set = set(static_feature[static_feature.num >= 10].index)

    # calculate 10 nearest neighborhoods for each station, sort by distance and store their index, get a map
    neighbor_distance_map = {}
    matrix_distance = np.load(exp_data_path + os.sep + 'similarity' + os.sep + 'similarity_distance_{}_numpy.npy'.format(city), allow_pickle=True)
    all_distance_map = {i: [] for i in range(station_count[city])}
    for i in range(station_count[city]):
        if i not in station_set:
            continue
        for j in range(station_count[city]):
            if j not in station_set:
                continue
            all_distance_map[i].append((j, matrix_distance[i][j]))
        all_distance_map[i].sort(key=lambda x : x[1], reverse=True)
        neighbor_distance_map[i] = [idx for idx, distance in all_distance_map[i][:10]]

    # 11 times header, get static neighborhood feature for each station(in S), get csv: neighbor_feature_{city}.csv
    ALL_HEADER = ['index']
    ALL_HEADER.extend(GENERAL_HEADER)
    for i in range(10):
        for j in GENERAL_HEADER:
            ALL_HEADER.append('{}_{}'.format(j, i))

    raw_data = np.empty((len(neighbor_distance_map), len(ALL_HEADER)))
    for i, idx in enumerate(neighbor_distance_map.keys()):
        raw_data[i][0] = idx
        raw_data[i][1:1+len(GENERAL_HEADER)] = static_feature.iloc[idx]['num':'mall']
        for j in range(10):
            neighbor_idx = neighbor_distance_map[idx][j]
            raw_data[i][1+len(GENERAL_HEADER)*(j+1):1+len(GENERAL_HEADER)*(j+2)] = static_feature.iloc[neighbor_idx]['num':'mall']
    neighbor_feature_data = pd.DataFrame(raw_data, columns=ALL_HEADER)
    print('neighbor feature')
    print(neighbor_feature_data)

    neighbor_feature_path = exp_data_path + os.sep + 'static' + os.sep + 'static_neighor_feature_{}.csv'.format(city)
    if os.path.exists(neighbor_feature_path):
        os.remove(neighbor_feature_path)
    neighbor_feature_data.to_csv(neighbor_feature_path)

    # create final csv(11 times header with basic info(time_index + time_embed_index))
    # if index in S, fill basic info, neighbor_feature and demand

    demand = np.load(exp_data_path + os.sep + 'station' + os.sep + 'demand_{}.npy'.format(city), allow_pickle=True)
    time_count = demand.shape[1]

    DEMAND_HEADER = []
    DEMAND_HEADER.extend(ALL_HEADER)
    DEMAND_HEADER.extend(['time_index', 'time_embed', 'demand'])
    neighbor_demand_raw_data = np.empty(((len(neighbor_distance_map)*time_count, len(DEMAND_HEADER))))

    # get time map like {"0800": 1, "0830": 2, ....}
    time_index_map = np.load(exp_data_path + os.sep + 'station_list' + os.sep + 'time_index.npy', allow_pickle=True)
    time_index_map = dict(time_index_map.tolist())
    time_map = {t: i for i, t in enumerate(sorted(set([k[-4:] for k in time_index_map['rev_index'].keys()])))}

    cur_idx = 0
    for time_idx in range(time_count):
        time_embed_idx = time_map[time_index_map['index'][time_idx][-4:]]
        for station_idx in station_set:
            neighbor_demand_raw_data[cur_idx][0:len(ALL_HEADER)] = neighbor_feature_data.loc[neighbor_feature_data['index']==station_idx, 'index':'mall_9']
            neighbor_demand_raw_data[cur_idx][len(ALL_HEADER)] = time_idx
            neighbor_demand_raw_data[cur_idx][len(ALL_HEADER)+1] = time_embed_idx
            neighbor_demand_raw_data[cur_idx][len(ALL_HEADER)+2] = demand[station_idx][time_idx][-1]
            # todo add slow demand and quick demand here
            cur_idx = cur_idx + 1
    print(cur_idx, neighbor_demand_raw_data.shape)

    neighbor_demand_data = pd.DataFrame(neighbor_demand_raw_data, columns=DEMAND_HEADER)
    print('neighbor demand')
    print(neighbor_demand_data)

    neighbor_demand_path = exp_data_path + os.sep + 'static' + os.sep + 'neighbor_demand_{}.csv'.format(city)
    if os.path.exists(neighbor_demand_path):
        os.remove(neighbor_demand_path)
    neighbor_demand_data.to_csv(neighbor_demand_path)


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

def semi_all_static_feature(city):
    """
    get semi_static_feature(add slow, fast, without near station, near stub)
    """
    poi_frequency = np.load(exp_data_path + os.sep + 'poi_frequency' + os.sep + 'poi_frequency_{}.npy'.format(city),
                            allow_pickle=True)  # .tolist()
    poi_num = np.load(exp_data_path + os.sep + 'poi' + os.sep + 'poi_{}.npy'.format(city), allow_pickle=True)
    poi_entropy = np.load(exp_data_path + os.sep + 'poi_entropy' + os.sep + 'poi_entropy_{}.npy'.format(city),
                          allow_pickle=True)
    road = np.load(exp_data_path + os.sep + 'roadnet' + os.sep + 'roadnet_{}.npy'.format(city), allow_pickle=True)
    transportation = np.load(exp_data_path + os.sep + 'transportation' + os.sep + 'transportation_{}.npy'.format(city),
                             allow_pickle=True)
    commerce = np.load(exp_data_path + os.sep + 'commerce' + os.sep + 'commerce_{}.npy'.format(city), allow_pickle=True)

    file_name = exp_data_path + os.sep + 'station' + os.sep + 'all_demand_{}.npy'.format(city)
    demand_data = np.load(file_name, allow_pickle=True)
    total_num = demand_data[:, 0, -2, np.newaxis]
    slow_num = demand_data[:, 0, 0, np.newaxis]
    fast_num = demand_data[:, 0, 2, np.newaxis]

    raw_data = np.concatenate((slow_num, fast_num, total_num, poi_frequency, poi_num, poi_entropy, road, transportation, commerce), axis=1)
    csv_data = pd.DataFrame(raw_data, columns=SEMI_GENERAL_HEADER)
    print(csv_data.shape)
    # print(csv_data.iloc[:, 2])

    file_path = exp_data_path + os.sep + 'static' + os.sep + 'semi_static_feature_{}.csv'.format(city)
    if os.path.exists(file_path):
        os.remove(file_path)
    csv_data.to_csv(file_path)
    pass

def full_all_static_neighbor_and_demand(city):
    """
    get full_static_neighbor_feature, full_neighbor_demand
    """
    # get station set S with more than 10 charge equipment
    static_file_path = exp_data_path + os.sep + 'static' + os.sep + 'semi_static_feature_{}.csv'.format(city)
    static_feature = pd.read_csv(static_file_path, header=0)
    station_set = set(static_feature[static_feature.total_num >= 3].index)

    # calculate 10 nearest neighborhoods for each station, sort by distance and store their index, get a map
    neighbor_distance_map = {}
    matrix_distance = np.load(
        exp_data_path + os.sep + 'similarity' + os.sep + 'similarity_distance_{}_numpy.npy'.format(city),
        allow_pickle=True)
    all_distance_map = {i: [] for i in range(station_count[city])}
    for i in range(station_count[city]):
        if i not in station_set:
            continue
        for j in range(station_count[city]):
            if j not in station_set:
                continue
            all_distance_map[i].append((j, matrix_distance[i][j]))
        all_distance_map[i].sort(key=lambda x: x[1], reverse=True)
        neighbor_distance_map[i] = [idx for idx, distance in all_distance_map[i][:10]]

    # todo using all_distance_map to get near station and near charge stub feature
    near_station_count = {}
    near_stub_count = {}
    for i, _ in neighbor_distance_map.items():
        less_three = 0
        near_stub_count[i] = 0
        for j, dis in all_distance_map[i]:
            if j == i:
                continue
            if dis > 1/3:
                less_three = less_three + 1
                near_stub_count[i] = near_stub_count[i] + static_feature.iloc[j, 2]
        near_station_count[i] = less_three

    # 11 times header, get static neighborhood feature for each station(in S), get csv: full_neighbor_feature_{city}.csv
    ALL_HEADER = ['index']
    ALL_HEADER.extend(FULL_GENERAL_HEADER)
    for i in range(10):
        for j in FULL_GENERAL_HEADER:
            ALL_HEADER.append('{}_{}'.format(j, i))

    raw_data = np.empty((len(neighbor_distance_map), len(ALL_HEADER)))
    for i, idx in enumerate(neighbor_distance_map.keys()):
        raw_data[i][0] = idx
        raw_data[i][1] = near_station_count[idx]
        raw_data[i][2] = near_stub_count[idx]
        raw_data[i][3:3 + len(SEMI_GENERAL_HEADER)] = static_feature.iloc[idx]['slow_num':'mall']
        for j in range(10):
            neighbor_idx = neighbor_distance_map[idx][j]
            raw_data[i][1 + len(FULL_GENERAL_HEADER) * (j + 1)] = near_station_count[neighbor_idx]
            raw_data[i][1 + len(FULL_GENERAL_HEADER) * (j + 1) + 1] = near_stub_count[neighbor_idx]
            raw_data[i][1 + len(FULL_GENERAL_HEADER) * (j + 1) + 2: 1 + len(FULL_GENERAL_HEADER) * (j + 2)] = static_feature.iloc[neighbor_idx]['slow_num':'mall']
    neighbor_feature_data = pd.DataFrame(raw_data, columns=ALL_HEADER)
    print('neighbor feature')
    print(neighbor_feature_data.shape)

    neighbor_feature_path = exp_data_path + os.sep + 'static' + os.sep + 'full_static_neighor_feature_{}.csv'.format(
        city)
    if os.path.exists(neighbor_feature_path):
        os.remove(neighbor_feature_path)
    neighbor_feature_data.to_csv(neighbor_feature_path)

    # create final csv(11 times header with basic info(time_index + time_embed_index))
    # if index in S, fill basic info, neighbor_feature and demand

    demand = np.load(exp_data_path + os.sep + 'station' + os.sep + 'demand_{}.npy'.format(city), allow_pickle=True)
    time_count = demand.shape[1]

    DEMAND_HEADER = []
    DEMAND_HEADER.extend(ALL_HEADER)
    DEMAND_HEADER.extend(['time_index', 'time_embed', 'slow_demand', 'fast_demand', 'total_demand'])
    neighbor_demand_raw_data = np.empty(((len(neighbor_distance_map) * time_count, len(DEMAND_HEADER))))

    # get time map like {"0800": 1, "0830": 2, ....}
    time_index_map = np.load(exp_data_path + os.sep + 'station_list' + os.sep + 'time_index.npy', allow_pickle=True)
    time_index_map = dict(time_index_map.tolist())
    time_map = {t: i for i, t in enumerate(sorted(set([k[-4:] for k in time_index_map['rev_index'].keys()])))}

    cur_idx = 0
    for time_idx in range(time_count):
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
    print(cur_idx, neighbor_demand_raw_data.shape)

    neighbor_demand_data = pd.DataFrame(neighbor_demand_raw_data, columns=DEMAND_HEADER)
    print('neighbor demand')
    print(neighbor_demand_data)

    neighbor_demand_path = exp_data_path + os.sep + 'static' + os.sep + 'full_neighbor_demand_{}.csv'.format(city)
    if os.path.exists(neighbor_demand_path):
        os.remove(neighbor_demand_path)
    neighbor_demand_data.to_csv(neighbor_demand_path)

# unused
def make_cdp_gcn_matrix(city):
    # get id map from neighbor feature
    neighbor_feature_path = exp_data_path + os.sep + 'static' + os.sep + 'static_neighor_feature_{}.csv'.format(city)
    ALL_HEADER = ['index']
    ALL_HEADER.extend(GENERAL_HEADER)
    for i in range(10):
        for j in GENERAL_HEADER:
            ALL_HEADER.append('{}_{}'.format(j, i))

    neighbor_feature_data = pd.read_csv(neighbor_feature_path, usecols=ALL_HEADER)
    # print(neighbor_feature_data)
    id_map = {int(idx) : i for i, idx in enumerate(neighbor_feature_data['index'])}
    # save
    map_path = exp_data_path + os.sep + 'static' + os.sep + 'id_map_{}.file'
    if os.path.exists(map_path):
        os.remove(map_path)
    with open(map_path, "wb") as f:
        pickle.dump(id_map, f)

    # fill the poi feature
    raw_data = np.empty((len(id_map), 13))
    for old_idx, new_idx in id_map.items():
        raw_data[new_idx] = neighbor_feature_data.loc[neighbor_feature_data['index']==old_idx, 'hospital_f_0':'fast_food_f_0']

    # calculate the poi matrix and save
    similarity_matrix = np.corrcoef(raw_data) * 0.5 + 0.5
    print(similarity_matrix.shape)
    file_name = exp_data_path + os.sep + 'static' + os.sep + 'similarity_poi_{}'.format(city)
    if os.path.exists(file_name):
        os.remove(file_name)
    np.save(file_name, similarity_matrix)
    pass


if __name__ == '__main__':
    time_index_map = np.load(exp_data_path + os.sep + 'station_list' + os.sep + 'time_index.npy', allow_pickle=True)
    time_index_map = dict(time_index_map.tolist())
    time_map = {t: i for i, t in enumerate(sorted(set([k[-4:] for k in time_index_map['rev_index'].keys()])))}
    print(len(time_map))
    for city in data_length.keys():
        # 4.1-2
        # filter data < 0.9 and miss record > 0.1
        # print('filter for', city)
        # filter_demand(city, 0.9)

        # get top station
        # print('filter for', city)
        # get_top_station_set(city)

        # get new station set
        # get_top_station_csv(city)

        # 4.1-3
        # print('dispatch for', city)
        # get_city_weather_and_dispatch(city)
        # weather_of_weather_type(city)
        # weather_of_temperature(city)
        # weather_of_wind(city)
        # weather_of_air(city)

        # 4.1-4
        # print('make demand tensor for', city)
        # filter_station(city)
        # make_demand_tensor(city)

        # 4.1-5
        # print('filter road net for', city)
        # filter_road_and_poi_and_save_npy(city, 'roadnet', float)

        # 4.1-6
        # print('filter for poi related data for', city)
        # make_poi_frequency_and_entropy(city)
        # filter_road_and_poi_and_save_npy(city, 'poi', int)
        # filter_road_and_poi_and_save_npy(city, 'poi_frequency', float)
        # filter_road_and_poi_and_save_npy(city, 'poi_entropy', float)
        # filter_road_and_poi_and_save_npy(city, 'commerce', int)
        # filter_road_and_poi_and_save_npy(city, 'transportation', int)

        # 4.2-1
        # print('make poi similarity matrix for', city)
        # make_poi_frequency_and_road_similarity_matrix(city, 'poi_frequency')

        # 4.2-2
        # print('make roadnet similarity matrix for', city)
        # make_poi_frequency_and_road_similarity_matrix(city, 'roadnet')

        # 4.2-3
        # print('make distance similarity matrix for', city)
        # make_distance_matrix(city)

        # average demand
        # print('make average demand for', city)
        # fix_demand(city)
        # make_avg_demand(city)

        # make cdp demand
        # print('make cdp demand for', city)
        # get_all_station_feature(city)
        # get_neigh_demand(city)

        # get full data
        # print('get full static feature and demand for', city)
        # semi_all_static_feature(city)
        # full_all_static_neighbor_and_demand(city)
        pass
    # get_time_index()
    # weather_of_hour()
    # weather_of_day()
    # holiday()
    pass
