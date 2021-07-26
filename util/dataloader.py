import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from preprocess.process import GENERAL_HEADER
import h5py
import time
import math

import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')

logger = logging.getLogger(__name__)

class CdpDataLoader(Dataset):
    def __init__(self, x, t, y):
        self.x = x
        self.t = t
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.t[idx], self.y[idx]


def get_data_loader(source_city, target_city, batch_size, num_neighbor, train_mode, window=False):
    start_time = time.time()
    if window:
        h5_path = 'data/exp_data/static/h5'
    else:
        h5_path = r'/home/ryj/renyajie/exp/NETL/data/exp_data/static/h5'

    file_name = '{}_{}_{}_{}.h5'.format(train_mode, num_neighbor, source_city, target_city) if train_mode != 'target' else 'target_{}_{}.h5'.format(num_neighbor, target_city)
    file_path = h5_path + os.sep + file_name
    if not os.path.exists(file_path):
        logger.info('{} does not exists, going to get'.format(file_name))
        get_data(source_city, target_city, num_neighbor, train_mode, window)

    data_loaders = {}
    f = h5py.File(file_path, 'r')
    # common data
    source_train_x = f['source_train_x']
    source_train_t = f['source_train_t']
    source_train_y = f['source_train_y']
    target_train_x = f['target_train_x']
    target_train_t = f['target_train_t']
    target_train_y = f['target_train_y']
    data_loaders['combine_source'] = DataLoader(CdpDataLoader(source_train_x, source_train_t, source_train_y),
                                                batch_size=batch_size // 2, shuffle=True, drop_last=True)
    data_loaders['combine_target'] = DataLoader(CdpDataLoader(target_train_x, target_train_t, target_train_y),
                                                batch_size=batch_size // 2, shuffle=True, drop_last=True)
    if train_mode == 'combine':
        val_x = f['val_x']
        val_t = f['val_t']
        val_y = f['val_y']
        test_x = f['test_x']
        test_t = f['test_t']
        test_y = f['test_y']
        data_loaders['val'] = DataLoader(CdpDataLoader(val_x, val_t, val_y), batch_size=batch_size, shuffle=True, drop_last=True)
        data_loaders['test'] = DataLoader(CdpDataLoader(test_x, test_t, test_y), batch_size=batch_size, shuffle=True, drop_last=True)
    else:
        train_x = f['train_x']
        train_t = f['train_t']
        train_y = f['train_y']
        val_x = f['val_x']
        val_t = f['val_t']
        val_y = f['val_y']
        test_x = f['test_x']
        test_t = f['test_t']
        test_y = f['test_y']
        data_loaders['train'] = DataLoader(CdpDataLoader(train_x, train_t, train_y), batch_size=batch_size, shuffle=True, drop_last=True)
        data_loaders['val'] = DataLoader(CdpDataLoader(val_x, val_t, val_y), batch_size=batch_size, shuffle=True, drop_last=True)
        data_loaders['test'] = DataLoader(CdpDataLoader(test_x, test_t, test_y), batch_size=batch_size, shuffle=True, drop_last=True)

    end_time = time.time()
    logger.info('get data loader cost {} s'.format(end_time - start_time))
    return data_loaders

def get_data(source_city, target_city, num_neighbor, train_mode, window = False):
    """
    make data loader for model training
    :return: a map store different data loader
    """
    start_time = time.time()
    if window:
        neighbor_demand_path = 'E:\\code\\netl\\data\\exp_data\\static'
    else:
        neighbor_demand_path = '/home/ryj/renyajie/exp/NETL/data/exp_data/static'

    if window:
        h5_path = 'E:\\code\\netl\\data\\exp_data\\static\\h5'
    else:
        h5_path = r'/home/ryj/renyajie/exp/NETL/data/exp_data/static/h5'

    feature_names = []
    for i in range(num_neighbor):
        feature_names += [('{}_{}').format(x, i) for x in GENERAL_HEADER]

    # load data for source city and target city
    source_df = pd.read_csv(neighbor_demand_path + os.sep + 'neighbor_demand_{}.csv'.format(source_city))
    logger.info('read source data, len is {}'.format(source_df.shape[0]))
    target_df = pd.read_csv(neighbor_demand_path + os.sep + 'neighbor_demand_{}.csv'.format(target_city))
    logger.info('read target data, len is {}'.format(target_df.shape[0]))

    # divide source city for train data and validate data
    source_train, source_val = train_test_split(source_df, test_size=0.2)
    logger.info('source divide')
    logger.info('source_train len is {}'.format(source_train.shape[0]))
    logger.info('source_val len is {}'.format(source_val.shape[0]))

    # align data
    target_train = target_df.copy(deep=True)
    if source_train.shape[0] < target_train.shape[0]:
        target_train = target_train.iloc[0:source_train.shape[0]]
        logger.info('align the source data, len is {}'.format(source_train.shape[0]))
    elif source_train.shape[0] > target_df.shape[0]:
        source_train = source_train.iloc[0:target_train.shape[0]]
        logger.info('align the target data, len is {}'.format(target_train.shape[0]))

    # get combine_source and combine target
    source_train_x = source_train[feature_names].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    source_train_t = source_train['time_embed']
    source_train_y = source_train['demand']
    logger.info('get source train data')

    target_train_x = target_train[feature_names].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    target_train_t = target_train['time_embed']
    target_train_y = target_train['demand']
    logger.info('get target train data')

    # make dann data, align the train data length
    if train_mode == 'combine':

        # get val
        val_x = source_val[feature_names].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
        val_t = source_val['time_embed']
        val_y = source_val['demand']
        logger.info('get val data')

        # get test
        test_x = target_df[feature_names].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
        test_t = target_df['time_embed']
        test_y = target_df['demand']
        logger.info('get test data')

        data_map = {'source_train_x': source_train_x, 'source_train_t': source_train_t, 'source_train_y': source_train_y,
                    'target_train_x': target_train_x, 'target_train_t': target_train_t, 'target_train_y': target_train_y,
                    'val_x': val_x, 'val_t': val_t, 'val_y': val_y, 'test_x': test_x, 'test_t': test_t, 'test_y': test_y}
    # make target data
    elif train_mode == 'target':
        target_train_data, target_test = train_test_split(target_df, test_size=0.2)
        target_train, target_val = train_test_split(target_train_data, test_size=0.1)
        logger.info('divide data into train val and test')

        train_x = target_train[feature_names].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
        train_t = target_train['time_embed']
        train_y = target_train['demand']
        logger.info('get train data')

        val_x = target_val[feature_names].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
        val_t = target_val['time_embed']
        val_y = target_val['demand']
        logger.info('get val data')

        test_x = target_test[feature_names].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
        test_t = target_test['time_embed']
        test_y = target_test['demand']
        logger.info('get test data')

        data_map = {'train_x': train_x, 'train_t': train_t, 'train_y': train_y, 'val_x': val_x,
                    'val_t': val_t, 'val_y': val_y, 'test_x': test_x, 'test_t': test_t, 'test_y': test_y,
                    'source_train_x': source_train_x, 'source_train_t': source_train_t, 'source_train_y': source_train_y,
                    'target_train_x': target_train_x, 'target_train_t': target_train_t, 'target_train_y': target_train_y,
                    }

    else: # make source data
        source_train, source_val = train_test_split(source_df, test_size=0.2)
        logger.info('divide data into train val and test')

        train_x = source_train[feature_names].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
        train_t = source_train['time_embed']
        train_y = source_train['demand']
        logger.info('get train data')

        val_x = source_val[feature_names].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
        val_t = source_val['time_embed']
        val_y = source_val['demand']
        logger.info('get val data')

        test_x = target_df[feature_names].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
        test_t = target_df['time_embed']
        test_y = target_df['demand']
        logger.info('get test data')

        data_map = {'train_x': train_x, 'train_t': train_t, 'train_y': train_y, 'val_x': val_x,
                    'val_t': val_t, 'val_y': val_y, 'test_x': test_x, 'test_t': test_t, 'test_y': test_y,
                    'source_train_x': source_train_x, 'source_train_t': source_train_t, 'source_train_y': source_train_y,
                    'target_train_x': target_train_x, 'target_train_t': target_train_t, 'target_train_y': target_train_y,
                    }


    # cache data
    file_name = '{}_{}_{}_{}.h5'.format(train_mode, num_neighbor, source_city, target_city) if train_mode != 'target' else 'target_{}_{}.h5'.format(num_neighbor, target_city)
    f = h5py.File(h5_path + os.sep + file_name, 'w')
    for k, v in data_map.items():
        v = v.fillna(0)
        if k.endswith("_t") or k.endswith("_y"):
            f[k] = v.values.reshape((len(v), 1))
        else:
            f[k] = v.values.reshape((len(v), num_neighbor, -1))

    f.close()
    logger.info('cache these data')

    end_time = time.time()
    logger.info('get and cache data costs {} s'.format(end_time - start_time))


if __name__ == '__main__':
    get_data_loader('guangzhou', 'tianjing', 64, 1, 'combine')
    """
    for source_city in ['beijing', 'guangzhou']:
        for target_city in ['beijing', 'guangzhou', 'tianjing']:
            if source_city == target_city:
                continue
            for train_mode in ['combine', 'source', 'target']:
                    get_data_loader(source_city, target_city, 64, 10, train_mode)
                    logger.info('-' * 30)
    """
