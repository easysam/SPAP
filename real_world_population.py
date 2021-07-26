import os
import yaml
import argparse
import pandas as pd
import numpy as np
from random import randrange
from model.portable_stdann import CityTransfer
from util.log import build_log
from evaluation import revenue


def pop_planning(_budget, _demand, _cost):
    # ensure _demand and _cost have same shape
    if len(_demand) != len(_cost):
        raise RuntimeError
    # allocate budget in proportion to population
    alloc = _budget * (_demand / _demand.sum())
    # calculate charger count for each station
    plan = np.empty_like(_cost, dtype=int)
    remain = np.empty_like(_cost, dtype=int)
    for i in range(len(_demand)):
        plan[i] = alloc[i] / 2 // _cost[i]
        remain[i] = alloc[i] / 2 - plan[i] * _cost[i]
    # spend remaining budget
    remain_budget = remain.sum()
    remain_slow_budget = remain_budget / 2
    remain_fast_budget = remain_budget / 2
    while remain_slow_budget > _cost[:, 0].min():
        i = randrange(len(_demand))
        if remain_slow_budget >= _cost[i, 0]:
            plan[i, 0] += 1
            remain_slow_budget -= _cost[i, 0]
    while remain_fast_budget > _cost[:, 0].min():
        i = randrange(len(_demand))
        if remain_fast_budget >= _cost[i, 0]:
            plan[i, 0] += 1
            remain_fast_budget -= _cost[i, 0]
    print((plan * _cost).sum())
    return plan


if __name__ == '__main__':
    # Package configuration
    pd.set_option('display.width', 128)
    pd.set_option('display.max_columns', 8)
    # Arguments
    parser = argparse.ArgumentParser(description="Population-based planning")
    parser.add_argument('--source', type=str, required=False,
                        choices=['beijing', 'tianjing', 'guangzhou'], help='source city')
    parser.add_argument('--target', type=str, required=True,
                        choices=['beijing', 'tianjing', 'guangzhou'], help='target city')
    parser.add_argument('--gpu', type=str, default="0", help='use which gpu 0, 1 or 2')
    args = parser.parse_args()
    params = {"beijing_tianjing":
                  {"alpha": 0.3, "beta": 0.1, "neigh": 3, "bs": 64, "lr": 0.001, "dropout": 0.1, "epoch": 10},
              "beijing_guangzhou":
                  {"alpha": 0.3, "beta": 0.1, "neigh": 3, "bs": 64, "lr": 0.001, "dropout": 0.1, "epoch": 10},
              "guangzhou_tianjing":
                  {"alpha": 0.3, "beta": 0.1, "neigh": 3, "bs": 64, "lr": 0.001, "dropout": 0.1, "epoch": 10}}
    param = params["{}_{}".format(args.source, args.target)]
    candidate_nums = {"beijing": 137, "tianjing": 101, "guangzhou": 123}
    candidate_num = candidate_nums[args.target]
    df = pd.read_csv("data/exp_data/station_list/list_{}.csv".format(args.target), header=None,
                     names=["idx", "lat", "lng", "id"])
    # sample = center_subset(df, candidate_nums[args.target])
    # Project configuration
    with open('conf/file.yaml', 'r', encoding='utf-8') as f:
        file_path_conf = yaml.load(f.read(), Loader=yaml.FullLoader)
    logger = build_log("TIO", os.path.join(file_path_conf['log']['model_log'],
                                           "{}_{}_{}".format("BF", args.source, args.target)), need_console=True)

    demand = pd.read_csv(f'data/exp_data/station/population/{args.target}_pop_demand.csv')
    result = np.zeros(10, dtype=float)
    CO_t = np.array([[33, 54]] * candidate_num, dtype=int)
    P_t = np.array([[5.6, 48]] * candidate_num, dtype=float)
    budget = {'guangzhou': 80154, 'tianjing': 38763}[args.target]
    for single_run in range(10):
        C_t = pop_planning(budget, demand.to_numpy(), CO_t)
        city_transfer = CityTransfer(logger, args.source, args.target, gpu=args.gpu, param=param, C_t=C_t)
        city_transfer.fit()
        pred = city_transfer.predictor(C_t)
        time_key_cnt = len(pred) // candidate_nums[args.target]
        pred = pred.reshape(time_key_cnt, candidate_nums[args.target], -1).mean(axis=0)
        pred = pred[:, [0, 1]]
        R = revenue(C_t, pred, P_t)
        result[single_run] = R
        print(result[single_run])
    print(result)
