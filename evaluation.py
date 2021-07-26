import os
import numpy as np
import pandas as pd
from util.log import build_log
from preprocess.load_data import read_yaml
from model.portable_stdann import CityTransfer
import argparse


def revenue(C_t, Y_t, P_t):
    return (C_t * Y_t * P_t).sum()


def tuning_tensor(C_t, dim, increase=True):
    tuning_C = []
    for o_dim in range(C_t.shape[1]):
        if o_dim != dim:
            tuning_C.append(C_t[:, dim])
        else:
            if increase:
                tuning_C.append(C_t[:, o_dim] + 1)
            else:
                tuning_C.append(C_t[:, o_dim] - 1)
    return np.vstack(tuning_C).T


def extend(C_t):
    extend_Cs = []
    for dim in range(C_t.shape[1]):
        extend_Cs.append(tuning_tensor(C_t, dim, increase=True))
        extend_Cs.append(tuning_tensor(C_t, dim, increase=False))
    dim_increased = [C[:, np.newaxis, :] for C in extend_Cs]
    return np.concatenate(dim_increased, axis=1)


def Knapsack(weight, value, max_weight):
    import datetime
    starttime = datetime.datetime.now()
    last = [-1] * (max_weight + 1)
    last_s = [[]] * (max_weight + 1)
    if len(weight) == 0: return 0

    for i, (x, y) in enumerate(zip(weight[0], value[0])):
        if x > max_weight: continue
        if last[x] >= y:
            continue
        else:
            last[x] = y
            last_s[x] = [i]

    for i, (w, v) in enumerate(zip(weight[1:], value[1:]), start=1):
        current = [-1] * (max_weight + 1)
        current_s = [[]] * (max_weight + 1)
        for j, (x, y) in enumerate(zip(w, v)):
            for k in range(x, max_weight + 1):
                try:
                    if last[k - x] < 0: continue
                except IndexError:
                    print("IndexError in Knapsack:")
                    print(k)
                    print(x)
                    print(last.shape)
                    print(type(last))
                if current[k] >= last[k - x] + y:
                    continue
                else:
                    current[k] = last[k - x] + y
                    current_s[k] = last_s[k - x] + [j]
        last, current = current, last
        last_s, current_s = current_s, last_s
    endtime = datetime.datetime.now()
    print('DP-MCK time consumption: ', (endtime - starttime).seconds)
    return max(last), last_s[np.argmax(last)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TIO")
    # data setting
    parser.add_argument('--source', type=str, default='beijing', required=True,
                        choices=['beijing', 'tianjing', 'guangzhou'], help='source city')
    parser.add_argument('--target', type=str, default='tianjing', required=True,
                        choices=['beijing', 'tianjing', 'guangzhou'], help='target city')
    # model configuration
    parser.add_argument('--gpu', type=str, default="0", help='use which gpu 0, 1 or 2')
    parser.add_argument('--epoch', type=int, default=60, help='how many epoch to train')
    args = parser.parse_args()

    file_path_conf = read_yaml(windows=True)
    logger = build_log("TIO", os.path.join(file_path_conf['log']['model_log'], "{}_{}_{}"
        .format("TIO", args.source, args.target)), need_console=True)

    params = {"beijing_tianjing":
                 {"neigh": 1, "dropout": 0.1, "beta": 0.1, "alpha": 0.3, "bs": 64, "lr": 0.001, "epoch": 3},
             "beijing_guangzhou":
                 {"alpha": 0.3, "beta": 0.1, "neigh": 3, "bs": 64, "lr": 0.001, "dropout": 0.1, "epoch": 3},
             "guangzhou_tianjing":
                 {"alpha": 0.3, "beta": 0.1, "neigh": 3, "bs": 64, "lr": 0.001, "dropout": 0.1, "epoch": 3}}

    """Below is the Transfer Iteration Optimization (TIO) algorithm"""
    """Initialization algorithm input"""
    # budget unit is 1000 (k)
    param = params["{}_{}".format(args.source, args.target)]

    candidate_num = {"beijing": 137, "tianjing": 101, "guangzhou": 123}
    candidate_loc = candidate_num[args.target]

    # charging station distribution
    real_C_t = np.load("profiles_{}.npy".format(args.target)).astype(int)
    # cost for each type in each charging station
    CO_t = np.array([[33, 54]] * candidate_loc, dtype=int)
    # profit for each type in each charging station
    P_t = np.array([[5.6, 48]] * candidate_loc, dtype=float)
    # demand for each type in each charging station
    Y_t = np.zeros((candidate_loc, 2), dtype=float)

    budget = (real_C_t * CO_t).sum()
    print(budget)
    result = pd.DataFrame(columns=["even", "demand", "tio", "real"])

    budget = 20000
    for run_idx in range(1):
        budget += 5000
        result.loc[run_idx, "budget"] = budget
        """Core"""
        b_ = budget / (2 * candidate_loc)
        C_t = b_ // CO_t
        logger.warning("## Building CityTransfer")
        city_transfer = CityTransfer(logger,
                                     args.source, args.target,
                                     C_t,
                                     gpu=args.gpu, param=param)

        logger.warning("### TIO: Fitting for even budget allocation")
        city_transfer.fit()

        Y_t = city_transfer.predictor(C_t)
        # The predicted demand Y_t is t1_s1, t1_s2, ... ti_s1, ti_s2, then agg by station
        time_key_cnt = len(Y_t) // candidate_num[args.target]
        Y_t = Y_t.reshape(time_key_cnt, candidate_num[args.target], -1).mean(axis=0)
        Y_t = Y_t[:, [0, 1]]
        logger.info("C_t, Y_t, P_t:\n {},\n {},\n {}".format(C_t[:3], Y_t[:3], P_t[:3]))
        R_ = revenue(C_t, Y_t, P_t)

        result.loc[run_idx, "even"] = R_

        print("Initial revenue (even allocation) is: {}".format(R_))
        C_t_ = C_t

        Y_t_norm = Y_t / Y_t.sum()
        B_ = budget * Y_t_norm
        C_t = B_ // CO_t

        iteration_idx = 0
        while True:
            iteration_idx += 1

            logger.warning("### New iteration, {}".format(iteration_idx))
            logger.info("### Iter. {}: Update profiles using C_t".format(iteration_idx))
            city_transfer.update_target_profiles(C_t)
            if True or (city_transfer.discriminator(C_t) > 0.5):
                city_transfer.fit()
            C_t_extend = extend(C_t)
            Y_t_extend = city_transfer.extend_predictor(C_t_extend)

            # Y_t = Y_t_extend[:, 0]
            Y_t = city_transfer.predictor(C_t)
            Y_t = Y_t.reshape(time_key_cnt, candidate_num[args.target], -1).mean(axis=0)

            R = revenue(C_t, Y_t[:, [0, 1]], P_t)
            if R <= R_:
                print("Fitted revenue prediction in iter {} is {}".format(iteration_idx, R))
                if 1 == iteration_idx:
                    R_ = R
                else:
                    result.loc[run_idx, "tio"] = R_
                    break
            else:
                print("Higher fitted revenue prediction in iter {} is {}".format(iteration_idx, R))
                R_ = R
            if 1 == iteration_idx:
                result.loc[run_idx, "demand"] = R_

            C_t_ = C_t
            logger.info("### Invoke Knapsack algorithm. Input shape:")
            logger.info("C_t_extend: {}".format(C_t_extend.shape))
            Y_t_extend_agg = Y_t_extend.reshape(time_key_cnt, candidate_num[args.target], Y_t_extend.shape[1], -1).mean(axis=0)
            Y_t_extend_agg = Y_t_extend_agg[:, :, [0, 1]]
            logger.info("Y_t_extend_agg: {}".format(Y_t_extend_agg.shape))
            # Convert negative number to big positive number, avoiding negative weight and negative C_t result
            C_t_positive_weight = (C_t_extend < 0) * 1000 + C_t_extend
            pred_highest_R, arg_C_t = Knapsack((C_t_positive_weight * np.concatenate([CO_t[:, np.newaxis]]*4, axis=1)).sum(axis=2).astype(int),
                                               (C_t_extend * Y_t_extend_agg * np.concatenate([P_t[:, np.newaxis]]*4, axis=1)).sum(axis=2), budget)
            print("Knapsack profit prediction in iter {} is {}".format(iteration_idx, pred_highest_R))
            # print("New C_t index and distribution (sample for 5 stations):\n", arg_C_t)
            C_t = C_t_extend[np.arange(candidate_num[args.target]), arg_C_t]
            # print(C_t[:5])

        city_transfer.update_target_profiles(real_C_t)
        city_transfer.fit()
        real_y = city_transfer.predictor(real_C_t)
        # The predicted demand Y_t is t1_s1, t1_s2, ... ti_s1, ti_s2, then agg by station
        time_key_cnt = len(real_y) // candidate_num[args.target]
        real_y = real_y.reshape(time_key_cnt, candidate_num[args.target], -1).mean(axis=0)
        real_y = real_y[:, [0, 1]]
        logger.info("real_C_t, real_Y, P_t:\n {},\n {},\n {}".format(real_C_t[:3], real_y[:3], P_t[:3]))
        real_R = revenue(real_C_t, real_y, P_t)
        print("real R: ", real_R)

        result.loc[run_idx, "real"] = real_R

        result.to_csv("tio_scalable_result_{}_{}.csv".format(args.source, args.target), index=False, mode='a', header=False)
