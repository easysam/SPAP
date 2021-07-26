import os
import yaml
import time
import argparse
import pandas as pd
import numpy as np
from random import randrange
from tqdm import tqdm
from itertools import combinations
from sklearn.metrics.pairwise import haversine_distances
from model.portable_stdann import CityTransfer
from preprocess.load_data import read_yaml
from util.log import build_log
from evaluation import revenue, extend, Knapsack
from kdd18_cg import kdd18_cg


def bfs(budget, sample_idx, candidate_num, P_t):
    # 1. First brute force search for the optimal solution
    comb = combinations(range(budget + 2 * len(sample_idx) - 1), (2 * len(sample_idx) - 1))
    comb = np.array(list(comb))

    budget_plans = np.zeros((comb.shape[0], comb.shape[1] + 1))
    budget_plans[:, 0] = (comb[:, 0] + 1) - 1
    for i in range(1, comb.shape[1]):
        budget_plans[:, i] = comb[:, i] - comb[:, i - 1] - 1
    budget_plans[:, -1] = (budget + 2 * len(sample_idx) - 1) - comb[:, -1] - 1
    budget_plans = budget_plans.reshape((budget_plans.shape[0], len(sample_idx), 2))
    plans = np.unique(budget_plans // [2, 3], axis=0)
    with open("temp/unique_plans", "wb") as f:
        np.save(f, plans)
    plans = np.transpose(plans, axes=[1, 0, 2])
    print(plans)
    print(plans.shape)

    complete_plans = np.zeros((candidate_num, plans.shape[1], 2))
    complete_plans[sample_idx.to_numpy()] = plans
    print(complete_plans)

    result = city_transfer.extend_predictor(complete_plans)
    print(result.shape)
    repeat_P_t = np.expand_dims(P_t, axis=1).repeat(result.shape[1], axis=1)
    print(result.shape, repeat_P_t.shape)

    time_key_cnt = len(result) // candidate_num
    result_agg = result.reshape(time_key_cnt, candidate_num, result.shape[1], -1).mean(axis=0)
    result_agg = result_agg[:, :, [0, 1]]
    with open("temp/result_agg", "wb") as f:
        np.save(f, result_agg)
    print("Result:")
    print((complete_plans * result_agg * repeat_P_t))
    print((complete_plans * result_agg * repeat_P_t).sum(2).T.sum(1))
    bf_res = (complete_plans * result_agg * repeat_P_t).sum(2).T.sum(1).max()
    return bf_res


def center_subset(_df, _n, _lat='lat', _lng='lng'):
    stats = _df.describe()
    center = [[stats.loc["50%", _lat], stats.loc["50%", _lng]]]
    locations = _df[[_lat, _lng]].to_numpy()
    distances = haversine_distances(np.radians(center), np.radians(locations)).reshape(-1)
    chosen = np.argpartition(distances, _n)[:_n]
    _sample = _df.iloc[chosen]
    return _sample


if __name__ == '__main__':
    # Package configuration
    pd.set_option('display.width', 128)
    pd.set_option('display.max_columns', 8)
    # Arguments
    parser = argparse.ArgumentParser(description="brute-force search")
    parser.add_argument('--source', type=str, required=True,
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
    # Variable initialization
    EVEN_res = []
    TIO_res = []
    CG_res = []

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # 2. kdd18, cg algorithm
    # load demand for cg algorithm input
    all_demand = np.load("data/exp_data/station/demand_{}.npy".format(args.target), allow_pickle=True)
    Y_t = all_demand[:, :, :2].mean(axis=1)
    real_C_t = np.load("profiles_{}.npy".format(args.target)).astype(int)
    fast_charger_capacity = 8
    slow_charger_capacity = 1
    demand = Y_t * real_C_t * [slow_charger_capacity, fast_charger_capacity]
    result = np.zeros((25, 3, 10))
    result_path = file_path_conf['result']['planning']['g2t']
    for single_run in range(10):
        city_transfer = CityTransfer(logger, args.source, args.target, gpu=args.gpu, param=param, C_t=real_C_t)
        city_transfer.fit()
        for budget_idx, budget in enumerate(range(1000, 50001, 2000)):
            CO_t = np.array([[33, 54]] * candidate_num, dtype=int)
            P_t = np.array([[5.6, 48]] * candidate_num, dtype=float)
            C_t_cg = kdd18_cg(budget, candidate_num, CO_t, demand)
            print(C_t_cg)
            # city_transfer = CityTransfer(logger, args.source, args.target, gpu=args.gpu, param=param, C_t=complete_C_t)
            # city_transfer.fit()
            city_transfer.update_target_profiles(C_t_cg)
            Y_t = city_transfer.predictor(C_t_cg)
            time_key_cnt = len(Y_t) // candidate_num
            Y_t = Y_t.reshape(time_key_cnt, candidate_num, -1).mean(axis=0)
            # print(complete_C_t)
            # print(Y_t[:, :2])
            # print(P_t)
            R = revenue(C_t_cg, Y_t[:, :2], P_t)
            CG_res.append(R)
            result[budget_idx, 0, single_run] = R

            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            # 3. Next evaluate TIO
            P_t = np.array([[5.6, 48]] * candidate_num, dtype=float)
            CO_t = np.array([[33, 54]] * candidate_num, dtype=int)
            # 2.1 Build initial C_t
            # b_ = budget / (2 * candidate_num)
            # C_t = b_ // CO_t
            # remain_b = budget - (C_t * CO_t).sum()
            C_t = np.array([[0, 0]] * candidate_num, dtype=int)
            remaining_slow_b = budget / 2
            remaining_fast_b = budget / 2
            while remaining_slow_b > CO_t[:, 0].min():
                i = randrange(candidate_num)
                if remaining_slow_b >= CO_t[i, 0]:
                    C_t[i, 0] += 1
                    remaining_slow_b -= CO_t[i, 0]
            while remaining_fast_b > CO_t[:, 1].min():
                i = randrange(candidate_num)
                if remaining_fast_b >= CO_t[i, 1]:
                    C_t[i, 1] += 1
                    remaining_fast_b -= CO_t[i, 1]
            # remain_b = budget
            # while remain_b > CO_t.min():
            #     i = randrange(candidate_num)
            #     if remain_b > CO_t[i, 0]:
            #         C_t[i, 0] += 1
            #         remain_b -= CO_t[i, 0]
            #     if remain_b > CO_t[i, 1]:
            #         C_t[i, 1] += 1
            #         remain_b -= CO_t[i, 1]

            # 2.2 Prediction
            city_transfer.update_target_profiles(C_t)
            Y_t = city_transfer.predictor(C_t)
            time_key_cnt = len(Y_t) // candidate_num
            Y_t = Y_t.reshape(time_key_cnt, candidate_num, -1).mean(axis=0)

            R_ = revenue(C_t, Y_t[:, :2], P_t)
            print("Initial revenue (even allocation) is: {}".format(R_))
            EVEN_res.append(R_)
            result[budget_idx, 1, single_run] = R_
            iteration_idx = 0
            while True:
                iteration_idx += 1
                logger.warning("### New iteration, {}".format(iteration_idx))
                logger.info("### Iter. {}: Update profiles using C_t".format(iteration_idx))

                C_t_extend = extend(C_t)
                C_t_extend = np.concatenate([C_t[:, np.newaxis, :], C_t_extend], axis=1)

                city_transfer.update_target_profiles(C_t)
                Y_t_extend = city_transfer.extend_predictor(C_t_extend)
                Y_t_extend_agg = Y_t_extend.reshape(time_key_cnt, candidate_num, Y_t_extend.shape[1], -1).mean(axis=0)
                # Convert negative number to big positive number, avoiding negative weight and negative C_t result
                C_t_positive_weight = (C_t_extend < 0) * 1000 + C_t_extend
                pred_highest_R, arg_C_t = Knapsack(
                    (C_t_positive_weight * np.concatenate([CO_t[:, np.newaxis]] * 5, axis=1)).sum(axis=2).astype(int),
                    (C_t_extend * Y_t_extend_agg[:, :, :2] * np.concatenate([P_t[:, np.newaxis]] * 5, axis=1)).sum(axis=2), budget)
                print("Knapsack profit prediction in iter {} is {}".format(iteration_idx, pred_highest_R))
                # print("New C_t index and distribution (sample for 5 stations):\n", arg_C_t)
                C_t = C_t_extend[np.arange(candidate_num), arg_C_t]
                print("Below plan achieve {} revenue".format(pred_highest_R))
                print(C_t)
                if pred_highest_R <= R_:
                    TIO_res.append(R_)
                    result[budget_idx, 2, single_run] = R_
                    break
                else:
                    print("Higher fitted revenue prediction in iter {} is {}".format(iteration_idx, pred_highest_R))
                    R_ = pred_highest_R

            print(args)
            print(result)
        np.save(result_path, result)

