import os
import time
import argparse
import pandas as pd
import numpy as np
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
    print('Number of unique plan of {} budget: {}'.format(budget, plans.shape))
    # with open("temp/unique_plans", "wb") as f:
    #     np.save(f, plans)
    # plans = np.transpose(plans, axes=[1, 0, 2])
    # print(plans)
    # print(plans.shape)
    #
    # complete_plans = np.zeros((candidate_num, plans.shape[1], 2))
    # complete_plans[sample_idx.to_numpy()] = plans
    # print(complete_plans)
    #
    # result = city_transfer.extend_predictor(complete_plans)
    # print(result.shape)
    # repeat_P_t = np.expand_dims(P_t, axis=1).repeat(result.shape[1], axis=1)
    # print(result.shape, repeat_P_t.shape)
    #
    # time_key_cnt = len(result) // candidate_num
    # result_agg = result.reshape(time_key_cnt, candidate_num, result.shape[1], -1).mean(axis=0)
    # result_agg = result_agg[:, :, [0, 1]]
    # with open("temp/result_agg", "wb") as f:
    #     np.save(f, result_agg)
    # print("Result:")
    # print((complete_plans * result_agg * repeat_P_t))
    # print((complete_plans * result_agg * repeat_P_t).sum(2).T.sum(1))
    # bf_res = (complete_plans * result_agg * repeat_P_t).sum(2).T.sum(1).max()
    return 0
    return bf_res


if __name__ == '__main__':
    pd.set_option('display.width', 128)
    pd.set_option('display.max_columns', 8)

    parser = argparse.ArgumentParser(description="brute-force search")
    parser.add_argument('--source', type=str, required=True,
                        choices=['beijing', 'tianjing', 'guangzhou'], help='source city')
    parser.add_argument('--target', type=str, required=True,
                        choices=['beijing', 'tianjing', 'guangzhou'], help='target city')

    parser.add_argument('--gpu', type=str, default="0", help='use which gpu 0, 1 or 2')
    args = parser.parse_args()

    file_path_conf = read_yaml(windows=True)
    logger = build_log("TIO", os.path.join(file_path_conf['log']['model_log'], "{}_{}_{}" .format("BF", args.source, args.target)), need_console=True)
    BF_res = []
    TIO_res = []
    CG_res = []

    BF_time = []
    TIO_time = []
    CG_time = []
    for budget in range(11, 16, 2):
        candidate_nums = {"beijing": 137, "tianjing": 101, "guangzhou": 123}
        candidate_num = candidate_nums[args.target]

        params = {"beijing_tianjing":
                      {"alpha": 0.3, "beta": 0.1, "neigh": 3, "bs": 64, "lr": 0.001, "dropout": 0.1, "epoch": 3},
                  "beijing_guangzhou":
                      {"alpha": 0.3, "beta": 0.1, "neigh": 3, "bs": 64, "lr": 0.001, "dropout": 0.1, "epoch": 3},
                  "guangzhou_tianjing":
                      {"alpha": 0.3, "beta": 0.1, "neigh": 3, "bs": 64, "lr": 0.001, "dropout": 0.1, "epoch": 3}}
        param = params["{}_{}".format(args.source, args.target)]

        station_list = pd.read_csv("data/exp_data/station_list/list_{}.csv".format(args.target),
                                   header=None, index_col=0, names=['lat', 'lng', 'id'])
        # print(station_list.describe(percentiles=[0.45, 0.55]))
        sample = station_list.loc[(station_list['lat'] > 39.070156) &
                                  (station_list['lat'] < 39.119743) &
                                  (station_list['lng'] > 117.187130) &
                                  (station_list['lng'] < 117.336295)]

        print("used stations: \n", sample)
        exit(0)
        AVG_EARTH_RADIUS = 6371.0088
        locations = sample[["lat", "lng"]].to_numpy()
        dis_mat = haversine_distances(np.radians(locations)) * AVG_EARTH_RADIUS
        # print(dis_mat)

        CO_t = np.array([[2, 3]] * len(sample), dtype=int)
        P_t = np.array([[5.6, 48]] * candidate_num, dtype=float)

        # 0.1 Construct a CityTransfer to predict all possible plan
        city_transfer = CityTransfer(logger, args.source, args.target, gpu=args.gpu, param=param)
        city_transfer.fit()

        # 1. First brute force search for the optimal solution
        start = time.process_time()
        bf_res = bfs(budget, sample.index, candidate_num, P_t)
        time_consumption = time.process_time() - start
        BF_res.append(bf_res)
        BF_time.append(time_consumption)

        # 2. kdd18, cg algorithm
        # load demand for cg algorithm input
        # #################time#########################
        # start = time.process_time()
        # #################time#########################
        # all_demand = np.load("data/exp_data/station/demand_{}.npy".format(args.target), allow_pickle=True)
        # Y_t = all_demand[:, :, :2].mean(axis=1)
        # real_C_t = np.load("profiles_{}.npy".format(args.target)).astype(int)
        # fast_charger_capacity = 8
        # slow_charger_capacity = 1
        # demand = Y_t * real_C_t * [slow_charger_capacity, fast_charger_capacity]
        # C_t_cg = kdd18_cg(budget, len(sample), CO_t, demand)
        # complete_C_t = np.zeros((candidate_num, 2))
        # complete_C_t[sample.index.to_numpy()] = C_t_cg
        # Y_t = city_transfer.predictor(complete_C_t)
        # time_key_cnt = len(Y_t) // candidate_num
        # Y_t = Y_t.reshape(time_key_cnt, candidate_num, -1).mean(axis=0)
        # # print(complete_C_t)
        # # print(Y_t[:, :2])
        # # print(P_t)
        # R = revenue(complete_C_t, Y_t[:, :2], P_t)
        # #################time#########################
        # time_consumption = time.process_time() - start
        # CG_time.append(time_consumption)
        # #################time#########################
        # CG_res.append(R)

        # 3. Next evaluate TIO
        #################time#########################
        start = time.process_time()
        #################time#########################
        P_t = np.array([[5.6, 48]] * len(sample.index), dtype=float)
        CO_t = np.array([[2, 3]] * len(sample.index), dtype=int)
        # 2.1 Build initial C_t
        b_ = budget / (2 * len(sample))
        C_t = b_ // CO_t
        complete_C_t = np.zeros((candidate_num, 2))
        complete_C_t[sample.index.to_numpy()] = C_t
        # 2.2 Prediction
        Y_t = city_transfer.predictor(complete_C_t)
        time_key_cnt = len(Y_t) // candidate_num
        Y_t = Y_t.reshape(time_key_cnt, candidate_num, -1).mean(axis=0)
        Y_t = Y_t[sample.index, :2]

        R_ = revenue(C_t, Y_t, P_t)
        print("Initial revenue (even allocation) is: {}".format(R_))

        iteration_idx = 0
        while True:
            iteration_idx += 1
            logger.warning("### New iteration, {}".format(iteration_idx))
            logger.info("### Iter. {}: Update profiles using C_t".format(iteration_idx))

            C_t_extend = extend(C_t)
            C_t_extend = np.concatenate([C_t[:, np.newaxis, :], C_t_extend], axis=1)
            complete_C_t_extend = np.zeros((candidate_num, C_t_extend.shape[1], 2))
            complete_C_t_extend[sample.index, :] = C_t_extend

            Y_t_extend = city_transfer.extend_predictor(complete_C_t_extend)
            Y_t_extend_agg = Y_t_extend.reshape(time_key_cnt, candidate_num, Y_t_extend.shape[1], -1).mean(axis=0)
            Y_t_extend_agg = Y_t_extend_agg[sample.index, :, :2]
            logger.info("Y_t_extend_agg: {}".format(Y_t_extend_agg.shape))
            # Convert negative number to big positive number, avoiding negative weight and negative C_t result
            C_t_positive_weight = (C_t_extend < 0) * 1000 + C_t_extend
            pred_highest_R, arg_C_t = Knapsack(
                (C_t_positive_weight * np.concatenate([CO_t[:, np.newaxis]] * 5, axis=1)).sum(axis=2).astype(int),
                (C_t_extend * Y_t_extend_agg * np.concatenate([P_t[:, np.newaxis]] * 5, axis=1)).sum(axis=2), budget)
            print("Knapsack profit prediction in iter {} is {}".format(iteration_idx, pred_highest_R))
            # print("New C_t index and distribution (sample for 5 stations):\n", arg_C_t)
            C_t = C_t_extend[np.arange(len(sample)), arg_C_t]
            print("Below plan achieve {} revenue".format(pred_highest_R))
            print(C_t)
            if pred_highest_R <= R_:
                    TIO_res.append(R_)
                    break
            else:
                print("Higher fitted revenue prediction in iter {} is {}".format(iteration_idx, pred_highest_R))
                R_ = pred_highest_R
        #################time#########################
        time_consumption = time.process_time() - start
        TIO_time.append(time_consumption)
        #################time#########################
        print("budget: ", budget)
        print("bfs result: ", BF_res)
        print("TIO result: ", TIO_res)
        print("CG result: ", CG_res)
        print("bfs time: ", BF_time)
        print("TIO time: ", TIO_time)
        print("CG time: ", CG_time)

    print("bfs result: ", BF_res)
    print("TIO result: ", TIO_res)
    print("CG result: ", CG_res)
