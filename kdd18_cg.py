import numpy as np
from random import randrange
import argparse


def kdd18_cg(remaining_b, n, CO_t, demand, slow_charger_capacity=1, fast_charger_capacity=8):
    # C_cg is CG plan for target city
    demand = demand.copy()
    C_cg = np.zeros((n, 2), dtype=int)
    smallest_cost = CO_t.min()
    while remaining_b >= smallest_cost:
        max_reward = 0
        for i in range(n):
            reward_slow = min(demand[i, 0], slow_charger_capacity)
            if CO_t[i, 0] <= remaining_b and reward_slow > max_reward:
                max_reward = reward_slow
                select = (i, 0)
            reward_fast = min(demand[i, 1], fast_charger_capacity)
            if CO_t[i, 1] <= remaining_b and reward_fast > max_reward:
                max_reward = reward_fast
                select = (i, 1)
        if max_reward > 0:
            demand[select] -= max_reward
            C_cg[select] += 1
            remaining_b -= CO_t[select]
        else:
            break
    remaining_slow_b = remaining_b / 2
    remaining_fast_b = remaining_b / 2
    while remaining_slow_b > CO_t[:, 0].min():
        i = randrange(n)
        if remaining_slow_b >= CO_t[i, 0]:
            C_cg[i, 0] += 1
            remaining_slow_b -= CO_t[i, 0]
    while remaining_fast_b > CO_t[:, 1].min():
        i = randrange(n)
        if remaining_fast_b >= CO_t[i, 1]:
            C_cg[i, 1] += 1
            remaining_fast_b -= CO_t[i, 1]
    # while remaining_b >= smallest_cost:
    #     i = randrange(n)
    #     if remaining_b >= CO_t[i, 0]:
    #         C_cg[i, 0] += 1
    #         remaining_b -= CO_t[i, 0]
    #     if remaining_b >= CO_t[i, 1]:
    #         C_cg[i, 1] += 1
    #         remaining_b -= CO_t[i, 1]
    return C_cg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CG")
    # data setting
    parser.add_argument('--source', type=str, default='beijing', required=True,
                        choices=['beijing', 'tianjing', 'guangzhou'], help='source city')
    parser.add_argument('--target', type=str, default='tianjing', required=True,
                        choices=['beijing', 'tianjing', 'guangzhou'], help='target city')
    args = parser.parse_args()

    candidate_nums = {"beijing": 137, "tianjing": 101, "guangzhou": 123}
    candidate_num = candidate_nums[args.target]
    # charging station distribution
    real_C_t = np.load("profiles_{}.npy".format(args.target)).astype(int)
    # cost for each type in each charging station
    CO_t = np.array([[33, 54]] * candidate_num, dtype=int)
    # profit for each type in each charging station
    P_t = np.array([[5.6, 48]] * candidate_num, dtype=float)
    # demand for each type in each charging station, select slow, fast demand, take average in time dim
    all_demand = np.load("data/exp_data/station/demand_{}.npy".format(args.target), allow_pickle=True)
    Y_t = all_demand[:, :, :2].mean(axis=1)
    # budget
    budget = (real_C_t * CO_t).sum()

    # estimate slow and fast charging demand in each station
    # note that fast charger capacity is 8 times to slow charger
    fast_charger_capacity = 8
    slow_charger_capacity = 1
    demand = Y_t * real_C_t * [slow_charger_capacity, fast_charger_capacity]
    kdd18_cg(budget, candidate_num, CO_t, demand)
